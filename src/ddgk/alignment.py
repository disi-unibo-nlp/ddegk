import random
import os

from absl import app
from absl import flags

import warnings
warnings.filterwarnings('ignore')

FLAGS = flags.FLAGS
flags.DEFINE_string('data_set', None, 'The data set.')
flags.DEFINE_string('working_dir', None, 'The working directory.')
flags.DEFINE_float('node_embedding_coeff', 1, 'The coefficient for the loss related to node embeddings')
flags.DEFINE_float('node_label_coeff', 1, 'The coefficient for the loss related to node discrete labels')
flags.DEFINE_float('edge_label_coeff', 1, 'The coefficien or the loss related to edge labels')
flags.DEFINE_integer('source_index', None, 'The index of the source graph')
flags.DEFINE_integer('target_index', None, 'The index of the target graph')

from graph_embedding.ddgk import model
from graph_embedding.ddgk.common import load

def do_stuff(data_set, source_index, target_index, working_dir, node_embedding_coeff, node_label_coeff, edge_label_coeff):
  random.seed(42)

  print(f'Working in "{working_dir}"')

  graphs, num_edge_labels, num_node_labels = load(data_set)
  source = graphs[source_index]
  target = graphs[target_index]

  hparams = model.MutagHParams()
  hparams.node_embedding_loss_coefficient = node_embedding_coeff
  hparams.node_label_loss_coefficient = node_label_coeff
  hparams.incident_label_loss_coefficient = edge_label_coeff
  hparams.num_edge_labels = num_edge_labels
  hparams.num_node_labels = num_node_labels

  def ckpt(k):
    return os.path.join(working_dir, str(k), 'ckpt')

  # Encode the source
  os.mkdir(os.path.dirname(ckpt(source_index)))
  model.Encode(source, ckpt(source_index), hparams)

  _, source_probs = model.Score(source, target, ckpt(source_index), hparams, return_attention=True)
  return source, target, source_probs

def main(_):
  assert FLAGS.data_set
  assert FLAGS.source_index is not None
  assert FLAGS.target_index is not None
  assert os.path.isdir(FLAGS.working_dir)
  source, target, source_probs = do_stuff(FLAGS.data_set, FLAGS.source_index, FLAGS.target_index, FLAGS.working_dir, FLAGS.node_embedding_coeff, FLAGS.node_label_coeff, FLAGS.edge_label_coeff)
  print(len(source), len(target))
  print(source_probs)


if __name__ == '__main__':
    app.run(main)
