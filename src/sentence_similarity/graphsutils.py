from typing import Sequence, List
import json
import networkx as nx

DEFAULT_FILENAME = "graphs.json"

def load_graphs(filename: str) -> List[nx.Graph]:
    with open(filename) as graphs_file:
        content = json.load(graphs_file)
    return [nx.node_link_graph(g) for g in content]

def save_graphs(graphs: Sequence[nx.Graph], filename: str):
    with open(filename, 'w') as output_file:
        json.dump([nx.node_link_data(g) for g in graphs], output_file)

def get_embedding(graph: nx.Graph):
    return graph.graph['embedding'].items()
