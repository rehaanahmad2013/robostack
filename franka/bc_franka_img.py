from time import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import env_loader
import hydra
import numpy as np
import torch
import utils

from dm_env import specs
from logger import Logger
from buffers.replay_buffer import ReplayBufferStorage, make_replay_loader
from r2d2.trajectory_utils.misc import collect_trajectory, eval_trajectory

from video import TrainVideoRecorder, VideoRecorder
from agents import BCFrankaAgent

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    # output size of the convnet for 100x100 images
    # TODO: automate this?

    return BCFrankaAgent(obs_spec=obs_spec,
                         action_shape=action_spec.shape,
                         device=cfg.device,
                         lr=cfg.lr,
                         repr_dim=None,
                         feature_dim=cfg.feature_dim,
                         hidden_dim=cfg.hidden_dim,
                         use_tb=cfg.use_tb,
                         from_vision=cfg.from_vision)

class Workspace:
    def __init__(self, cfg, work_dir=None):
        if work_dir is None:
            self.work_dir = Path.cwd()
            print(f'New workspace: {self.work_dir}')
        else:
            self.work_dir = work_dir

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent,)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0 # how many episodes have been run

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env , self.eval_env, _, _, self.forward_demos, self.backward_demos = env_loader.make(self.cfg.env_name,
                                                                                                        action_repeat=self.cfg.action_repeat,
                                                                                                        num_frames=1,
                                                                                                        height=100,
                                                                                                        width=100)

        self.forward_demos = utils.refactor_demos_franka(self.forward_demos,
                                                         self.train_env._franka_transform_observation)
        if self.cfg.validation_split > 0.0:
            # sets the first K trajectories as validation dataset, other as train dataset
            self.forward_demos, self.val_demos = utils.split_demos_franka_shuf(self.forward_demos,
                                                                               split_val=self.cfg.validation_split)

            # HACK: data has been shuffled when splitting, observations and actions need to be re-aligned
            self.forward_demos['next_observations'][:-1] = self.forward_demos['observations'][1:]                                                                               

            # reformat for computation of validation loss
            len_val = self.val_demos['actions'].shape[0]

            self.val_set = {
                'images': np.array([self.val_demos['observations'][idx]['images'] for idx in range(len_val)]),
                'states': np.array([self.val_demos['observations'][idx]['state'] for idx in range(len_val)]),
                'actions': self.val_demos['actions'].copy()
            }
        self.val_set['states'] = self.val_set['states'].astype(np.float32)
        self.val_set['actions'] = self.val_set['actions'].astype(np.float32)
        self.val_set['images'] = self.val_set['images'].astype(np.float32)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.demo_buffer = ReplayBufferStorage(data_specs,
                                               self.work_dir / 'demo_buffer',)
        self.demo_loader = make_replay_loader(
            self.work_dir / 'demo_buffer', self.cfg.replay_buffer_size,
            self.cfg.agent.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, 1, 0.99)

        self._demo_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None, franka=True)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None, franka=True)

    @property
    def demo_iter(self):
        if self._demo_iter is None:
            self._demo_iter = iter(self.demo_loader)
        return self._demo_iter

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        if self.cfg.eval_mode == 'r2d2':
            print("beginning r2d2 collect trajectory fn")
            with torch.no_grad(), utils.eval_mode(self.agent):
                eval_trajectory(self.eval_env, policy=self.agent, measure_error=False, save_images=False, horizon=200)
        else:
            steps, episode, total_reward, episode_success = 0, 0, 0, 0
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                print(f'eval episode! {episode}')
                if self.cfg.save_video:
                    self.video_recorder.init(self.eval_env)
                episode_step = 0
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                uniform_action=False,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    if self.cfg.save_video:
                        self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    episode_step += 1
                    steps += 1

                episode += 1
                if self.cfg.save_video:
                    self.video_recorder.save(f'{self.global_frame}_ep_{episode}_.mp4')

            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('success_avg', episode_success / episode)
                log('episode_length', steps * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_steps, 1)
        eval_every_step = utils.Every(self.cfg.eval_and_log_every_steps, 1)

        self.demo_buffer.add_offline_data_franka(self.forward_demos)

        metrics = None
        while train_until_step(self.global_step):
            metrics = self.agent.update(demo_iter=self.demo_iter,
                                        step=self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')

            # log metrics and evaluate
            if eval_every_step(self.global_step):
                _, total_time = self.timer.reset()
                # compute validation loss here
                if self.cfg.validation_split > 0.0:
                    metrics = self.agent.compute_log_prob(obs_img=self.val_set['images'],
                                                          obs_state=self.val_set['states'],
                                                          actions=self.val_set['actions'])
                    self.logger.log_metrics(metrics, self.global_step, ty='train')

                # run self.logger metrics with eval
                with self.logger.log_and_dump_ctx(self.global_step,
                                                    ty='train') as log:
                    log('total_time', total_time)
                    log('step', self.global_step)

                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_step)

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot(step=self.global_step)

            self._global_step += 1

    def save_snapshot(self, step):
        # Add in info about the current timestep
        snapshot = self.work_dir / f'snapshot_{step}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step, work_dir=None):
        if work_dir is None:
            snapshot = self.work_dir / f'snapshot_{step}.pt'
        else:
            snapshot = work_dir / f'snapshot_{step}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

@hydra.main(config_path='../cfgs', config_name='bc_franka_img')
def main(cfg):
    if cfg.mode == 'train':
        work_dir_restore = None # add restore PATH here
        workspace = Workspace(cfg, work_dir=work_dir_restore)
        # workspace.load_snapshot(step=19000,
        #                         work_dir=work_dir_restore)
        workspace.train()

    elif cfg.mode == 'eval':
        print(f'evaluating the agent')
        cfg.use_tb = False
        workspace = Workspace(cfg)
        workspace.load_snapshot(step=cfg.eval_checkpoint_idx,
                                work_dir=Path(cfg.eval_dir))
        workspace.eval()


if __name__ == '__main__':
    main()
