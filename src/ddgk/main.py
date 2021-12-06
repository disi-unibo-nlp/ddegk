# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import collections
import itertools
import io
import multiprocessing
import random
import os
import sys
import math
import zipfile
import gzip
import json
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove annoying warnings

import argparse
import networkx as nx
import numpy as np
from scipy.spatial import distance
from six.moves import urllib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.ddgk import model
from src.ddgk.common import load

PROTO_CHOICE_RAND = 'random' # Choose num_sources random graphs
PROTO_CHOICE_TYPE = 'bytype' # Choose one graph for each event type (ignore num_sources)
PROTO_CHOICE_RAND_TYPE = 'randtype' # Choose num_sources random graphs, balanced from all event types
PROTO_CHOICE_VALUES = [PROTO_CHOICE_RAND, PROTO_CHOICE_TYPE, PROTO_CHOICE_RAND_TYPE]

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-prototypes', type=int, required=False, default=16)
  parser.add_argument('--node-embedding-coeff', type=float, required=True, help='The coefficient for the loss related to node embeddings')
  parser.add_argument('--node-label-coeff', type=float, required=True, help='The coefficient for the loss related to node discrete labels')
  parser.add_argument('--edge-label-coeff', type=float, required=True, help='The coefficien or the loss related to edge labels')
  parser.add_argument('--graphs-file', type=str, required=True, help='JSON file containing the graphs definitions')
  parser.add_argument('--working-dir-prefix', type=str, required=False, default='', help='Base directory where to create the working directory')
  parser.add_argument('--prototype-choice', type=str, required=True)
  parser.add_argument('--num-threads', default=32, type=int)
  args = parser.parse_args()
  args.working_dir = os.path.join(args.working_dir_prefix, f'SEMCOEFF_{args.node_embedding_coeff}_{args.node_label_coeff}_{args.edge_label_coeff}_data')
  assert args.num_prototypes > 0
  assert args.num_threads > 0
  if os.path.isdir(args.working_dir):
    choice = input(f"Directory '{args.working_dir}' already exists. Do you want to clean it? [y/N] ")
    if choice.lower() == 'y':
      print("Cleaning the directory")
      shutil.rmtree(args.working_dir)
    else:
      print("Aborting")
      sys.exit()
  os.mkdir(args.working_dir)
  return args

def choose_prototypes(graphs, args):
  print("Choosing prototypes")
  strategy = args.prototype_choice
  num_sources = args.num_prototypes
  by_type = dict()
  for i, g in graphs.items():
    by_type.setdefault(g.nodes[g.graph['root']]['type'], []).append(i)
  if strategy == PROTO_CHOICE_RAND:
    return dict(random.sample(graphs.items(), num_sources))
  elif strategy == PROTO_CHOICE_TYPE:
    indices = [max(of_type, key=lambda x: len(graphs[x])) for of_type in by_type.values()]
    return {i: graphs[i] for i in indices}
  elif strategy == PROTO_CHOICE_RAND_TYPE:
    candidates = list(itertools.chain.from_iterable(random.sample(of_type, math.ceil(num_sources / len(by_type))) for of_type in by_type.values()))
    return {i: graphs[i] for i in random.sample(candidates, num_sources)}
  else:
    raise Exception("Invalid prototype choice strategy")

def main():
  random.seed(42)
  args = parse_args()

  print(f'Working in "{args.working_dir}"')

  hparams = model.MutagHParams()
  hparams.node_embedding_loss_coefficient = args.node_embedding_coeff
  hparams.node_label_loss_coefficient = args.node_label_coeff
  hparams.incident_label_loss_coefficient = args.edge_label_coeff

  graphs, num_edge_labels, num_node_labels = load(args.graphs_file)
  hparams.num_edge_labels = num_edge_labels
  hparams.num_node_labels = num_node_labels

  sources = choose_prototypes(graphs, args)

  def ckpt(k):
    return os.path.join(args.working_dir, str(k), 'ckpt')

  with tqdm.tqdm(total=len(sources)) as pbar:
    tqdm.tqdm.write('Encoding {} source graphs...'.format(len(sources)))

    def encode(i):
      os.mkdir(os.path.dirname(ckpt(i)))
      model.Encode(sources[i], ckpt(i), hparams)
      pbar.update(1)

    pool = multiprocessing.pool.ThreadPool(args.num_threads)
    pool.map(encode, sources.keys())

  scores = collections.defaultdict(dict)

  output_file = open(os.path.join(args.working_dir, 'results.json'), 'w')
  with tqdm.tqdm(total=len(graphs) * len(sources), smoothing=0) as pbar:
    tqdm.tqdm.write('Scoring {} target graphs...'.format(len(graphs)))

    def score(i):
      for j, source in sources.items():
        scores[i][j] = model.Score(source, graphs[i], ckpt(j), hparams)[-1]
        pbar.update(1)

    for k in graphs.keys():
        score(k)
        output_file.write(json.dumps({'idx': k, 'embedding': scores[k]}) + '\n')
        output_file.flush()

  output_file.close()

if __name__ == '__main__':
  main()
