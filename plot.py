import argparse

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["text.usetex"] = True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Palatino"]})

import matplotlib.cm as cm
import matplotlib.ticker as ticker

import numpy as np
import re

import pickle as pkl
import os
import csv

# alpha = 0 -> no smoothing, alpha=1 -> perfectly smoothed to initial value
def smooth(x, alpha):
  if isinstance(x, list):
    size = len(x)
  else:
    size = x.shape[0]
  for idx in range(1, size):
    x[idx] = (1 - alpha) * x[idx] + alpha * x[idx - 1]
  return x

def make_graph_with_variance(vals,
                             x_interval,
                             max_index=int(1e8) ,
                             use_standard_error=True,
                             normalize_by_time=False):
  data_x = []
  data_y = []
  num_seeds = 0

  for y_coords, eval_interval in zip(vals, x_interval):
    num_seeds += 1
    data_y.append(smooth(y_coords, 0))
    x_coords = [eval_interval * idx for idx in range(len(y_coords))]
    data_x.append(x_coords)

  plot_dict = {}
  cur_max_index = max_index
  for cur_x, cur_y in zip(data_x, data_y):
    cur_max_index = min(cur_max_index, cur_x[-1])
    # print(cur_x[-1])
  print(cur_max_index)

  for cur_x, cur_y in zip(data_x, data_y):
    for x, y in zip(cur_x, cur_y):
      if normalize_by_time:
        y /= (x + x_interval)
      if x <= cur_max_index:
        if x in plot_dict.keys():
          plot_dict[x].append(y)
        else:
          plot_dict[x] = [y]

  print('output at step:', cur_max_index)
  print(np.mean(plot_dict[cur_max_index]), np.std(plot_dict[cur_max_index]) / np.sqrt(num_seeds))

  index, means, stds = [], [], []
  for key in sorted(plot_dict.keys()):  # pylint: disable=g-builtin-op
    index.append(key)
    means.append(np.mean(plot_dict[key]))
    if use_standard_error:
      stds.append(np.std(plot_dict[key]) / np.sqrt(num_seeds))
    else:
      stds.append(np.std(plot_dict[key]))

  means = np.array(smooth(means, 0.25))
  stds = np.array(smooth(stds, 0.25))

  return index, means, stds

def np_custom_load(fname):
  return np.load(fname).astype(np.float32)

def plotter(experiment_paths, mode, max_index=int(1e8), **plot_config):
  """Outermost function for plotting graphs with variance."""
  if mode == 'deployment':
    y_coords = [
        np_custom_load(os.path.join(experiment_path, 'deployed_eval.npy'))
        for experiment_path in experiment_paths
    ]
  elif mode == 'continuing':
    y_coords = [
        np_custom_load(os.path.join(experiment_path, 'continuing_eval.npy'))
        for experiment_path in experiment_paths
    ]

  eval_interval = [
        np_custom_load(os.path.join(experiment_path, 'eval_interval.npy'))
        for experiment_path in experiment_paths
  ]

  index, means, stds = make_graph_with_variance(y_coords,
                                                eval_interval,
                                                max_index=max_index,
                                                normalize_by_time=(mode=='continuing'))
  plt.plot(index, means, **plot_config)
  plt.fill_between(
      index, means - stds, means + stds, color=plot_config.get('color'), alpha=0.2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Plots for EARL evaluation')
  parser.add_argument('--eval_dir', type=str, default='../earl_benchmark/evaluation/benchmark_evaluation_numbers',
                      help='directory to load evaluation numbers and plots from')
  parser.add_argument('--env', type=str, default='tabletop',
                      help='environment name: [tabletop, door, peg, bulb, minitaur, kitchen]')
  parser.add_argument('--mode', type=str, default='deployment',
                      help='plot type: [deployment, continuing]')
  args = parser.parse_args()

  # basic configurations
  base_path = args.eval_dir
  plot_type = args.env
  mode = args.mode

  plot_config = {
    'VaPRL':
      {'color':'#73BA68', 'linestyle':'-', 'label':'VaPRL', 'linewidth':1.5},
    'FBRL':
      {'color':'r', 'linestyle':'-', 'label':'FBRL', 'linewidth':1.5},
    'naive':
      {'color':'c', 'linestyle':'-', 'label':'naive', 'linewidth':1.5},
    'R3L':
      {'color':'m', 'linestyle':'-', 'label':'R3L', 'linewidth':1.5},
    'oracle':
      {'color':'#9A9C99', 'linestyle':'-', 'label':'oracle', 'linewidth':1, 'dashes':[6, 6]},
    'MEDAL':
      {'color':'k', 'linestyle':'-', 'label':'state', 'linewidth':1.5},
    'MEDAL_ep':
      {'color':'k', 'linestyle':'-', 'label':'MEDAL_ep', 'linewidth':1.5},
    'MEDAL_vision':
      {'color':'g', 'linestyle':'-', 'label':'vision (+mods)', 'linewidth':1.5},
  }

  if plot_type == 'tabletop':
    max_index = int(1e6)
    plot_name = 'tabletop'
    title = 'Tabletop Organization'

    base_path = os.path.join(base_path, 'tabletop_organization')
    for method in ['MEDAL', 'MEDAL_vision']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue

      if method == 'MEDAL':
        folder_name = 'medal_multibuf_trunk'
        arl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/tabletop_manipulation', folder_name)
        experiments = []
        for path in os.listdir(arl_path):
          experiments.append(os.path.join(arl_path, path, next(os.walk(os.path.join(arl_path, path)))[1][0]))
      elif method == 'MEDAL_vision':
        folder_name = 'medal_vision/base'
        arl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/tabletop_manipulation', folder_name)
        experiments = []
        for path in os.listdir(arl_path):
          if path == '1':
            continue
          experiments.append(os.path.join(arl_path, path, next(os.walk(os.path.join(arl_path, path)))[1][0]))
      else:
        experiment_base = os.path.join(base_path, method.lower())
        experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])
    
  elif plot_type == 'peg':
    max_index = int(1e6)
    plot_name = 'sawyer_peg'
    title = 'Peg Insertion'

    base_path = os.path.join(base_path, 'sawyer_peg')
    for method in ['MEDAL', 'MEDAL_vision']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue

      if method == 'MEDAL':
        arl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/sawyer_peg', 'medal')
        experiments = []
        for path in os.listdir(arl_path):
          experiments.append(os.path.join(arl_path, path, next(os.walk(os.path.join(arl_path, path)))[1][0]))
      elif method == 'MEDAL_vision':
        folder_name = 'sawyer_peg/autonomous_bc_reg'
        arl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/medal_vice', folder_name)
        experiments = []
        for path in os.listdir(arl_path):
          experiments.append(os.path.join(arl_path, path, next(os.walk(os.path.join(arl_path, path)))[1][0]))
      else:
        experiment_base = os.path.join(base_path, method.lower())
        experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])
  
  elif plot_type == 'door':
    max_index = int(1e6)
    plot_name = 'sawyer_door'
    title = 'Door Closing'

    base_path = os.path.join(base_path, 'sawyer_door')
    for method in ['MEDAL', 'MEDAL_vision']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue
      
      if method in ['MEDAL']:
        torch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/sawyer_door/medal_uni_tradrb')
        experiments = []
        for path in os.listdir(torch_path):
          experiments.append(os.path.join(torch_path, path, next(os.walk(os.path.join(torch_path, path)))[1][0]))
      elif method in ['MEDAL_vision']:
        torch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/medal_vice/sawyer_door')
        experiments = []
        for path in os.listdir(torch_path):
          experiments.append(os.path.join(torch_path, path, next(os.walk(os.path.join(torch_path, path)))[1][0]))
      elif method in ['MEDAL_ep']:
        torch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_local/sawyer_door/medal_epbuffer')
        experiments = []
        for path in os.listdir(torch_path):
          experiments.append(os.path.join(torch_path, path, next(os.walk(os.path.join(torch_path, path)))[1][0]))
      else:
        experiment_base = os.path.join(base_path, method.lower())
        experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]

      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])

  elif plot_type == 'kitchen':
    max_index = int(5e6)
    plot_name = 'kitchen'
    title = 'kitchen'

    base_path = os.path.join(base_path, 'kitchen')
    for method in ['FBRL', 'naive', 'R3L', 'oracle']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue
      experiment_base = os.path.join(base_path, method.lower())
      experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])

  elif plot_type == 'minitaur':
    max_index = int(3e6)
    plot_name = 'minitaur'
    title = 'minitaur'

    base_path = os.path.join(base_path, 'minitaur_pen')
    for method in ['FBRL', 'naive', 'R3L', 'oracle']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue
      experiment_base = os.path.join(base_path, method.lower())
      experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])

  elif plot_type == 'bulb':
    max_index = int(5e6)
    plot_name = 'dhand_bulb'
    title = 'dhand bulb pickup'

    base_path = os.path.join(base_path, 'dhand_lightbulb')
    for method in ['FBRL', 'naive', 'oracle', 'R3L']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue
      experiment_base = os.path.join(base_path, method.lower())
      experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])

  # final plot config
  plt.grid(False)
  plt.legend(prop={'size': 12}, loc='upper left')

  # fig = legend.figure
  # fig.canvas.draw()
  # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  # fig.savefig(os.path.join(base_path, 'legend.png'), dpi=600, bbox_inches='tight')
  # exit()

  if mode == 'deployment':
    plot_name += '_transfer.png'
  elif mode == 'continuing':
    plot_name += '_ll.png'

  ax = plt.gca()
  # ax.set_ylim([0.0, 0.5])
  # plt.xlabel('Steps in Training Environment', fontsize=18)
  if mode == 'deployment':
    plt.ylabel('Deployed Policy Evaluation', fontsize=18)
  elif mode == 'continuing':
    plt.ylabel('Continuing Policy Evaluation', fontsize=18)

  plt.title(title, fontsize=18)
  print(list(ax.get_xticks()))
  # ax.set_xticks(list(ax.get_xticks())[1:-1])
  ax.set_yticks(list(ax.get_yticks())[2:-1])
  @ticker.FuncFormatter
  def major_formatter(x, pos):
    return '{:.1e}'.format(x).replace('+0', '')

  ax.xaxis.set_major_formatter(major_formatter)
  if mode == 'continuing':
    ax.yaxis.set_major_formatter(major_formatter)
  plt.setp(ax.get_xticklabels(), fontsize=10)
  plt.setp(ax.get_yticklabels(), fontsize=10)
  plt.savefig(os.path.join(base_path, plot_name), dpi=600, bbox_inches='tight')
