import networkx as nx
import json
import gzip

def load(data_set):
  print("Loading dataset...")
  if data_set.endswith('.gz'):
    print("Decompressing...")
    dataset_file = gzip.open(data_set, 'rt')
  else:
    dataset_file = open(data_set, 'r')
  data_set = json.load(dataset_file)
  dataset_file.close()
  graphs = [nx.node_link_graph(g) for g in data_set]

  # Normalize nodes ids
  for graph in graphs:
    mapping = {n: i for i,n in enumerate(graph.nodes)}
    nx.relabel_nodes(graph, mapping, copy=False)
    graph.graph['root'] = mapping[graph.graph['root']]

  # Normalize node labels (type)
  all_node_labels = sorted(set(t for g in graphs for _,t in g.nodes.data('type')))
  node_labels_mapping = {t: i for i,t in enumerate(all_node_labels)}
  for graph in graphs:
    for n,t in graph.nodes.data('type'):
      graph.nodes[n]['label'] = node_labels_mapping[t]

  # Normalize edge labels
  all_edge_labels = sorted(set(k for g in graphs for _,_,k in g.edges.data('key')))
  edge_labels_mapping = {k: i for i,k in enumerate(all_edge_labels)}
  for graph in graphs:
    for u,v,k in graph.edges.data('key'):
      graph.edges[u,v]['label'] = edge_labels_mapping[k]
  return {i: g for i,g in enumerate(graphs)}, len(edge_labels_mapping), len(node_labels_mapping)

