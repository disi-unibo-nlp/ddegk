from typing import Dict, List
import json
import networkx as nx
import datautils
import graphsutils
import numpy as np
from utils import get_by_indices
from datautils import DatasetT
from tqdm import tqdm

import torch
from torch import nn

SENTENCE_EMB_FILENAME = 'sentences_emb.json'
OUTPUT_FILENAME = 'similarities.json'

class SimilarityModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1000),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.ReLU(),
            # nn.Linear(1000,1000),
            nn.Sigmoid(),
            nn.Linear(1000, 1),
        )
    def forward(self, x):
        return self.layers(x)


def sent_id(graph: nx.Graph) -> int:
    return int(graph.graph['source_doc'][len('sent'):])

def load_graph_embeddings() -> Dict[int, np.ndarray]:
    graphs = graphsutils.load_graphs(graphsutils.DEFAULT_FILENAME)
    with open('graphs_emb.json') as input_file:
        graph_embs = list(map(json.loads, input_file))
    for emb in graph_embs:
        graphs[emb['idx']].graph['embedding'] = np.array([x[1] for x in sorted(emb['embedding'].items())])
    print("Merging graph embeddings by source sentence")
    return calculate_embeddings_from_graphs(graphs)

def calculate_embeddings_from_graphs(graphs) -> Dict[int, np.ndarray]:
    # Calc average
    reduced_graphs_embs: Dict[int, np.ndarray] = {}
    reduced_graphs_embs_counts: Dict[int, int] = {}
    for graph in graphs:
        sid = sent_id(graph)
        if sid in reduced_graphs_embs:
            reduced_graphs_embs[sid] += graph.graph['embedding']
            reduced_graphs_embs_counts[sid] += 1
        else:
            reduced_graphs_embs[sid] = graph.graph['embedding']
            reduced_graphs_embs_counts[sid] = 1
    for sid in reduced_graphs_embs:
        reduced_graphs_embs[sid] /= reduced_graphs_embs_counts[sid]
    return reduced_graphs_embs

def load_sentence_embeddings():
    with open(SENTENCE_EMB_FILENAME) as input_file:
        return {int(k): np.array(v) for k,v in json.load(input_file).items()} # JSON keys are always strings

def save_output(sent_similarities, graph_similarities, enhanced_similarities):
    with open(OUTPUT_FILENAME, 'w') as output_file:
        json.dump({'sentences': sent_similarities, 'graphs': graph_similarities, 'enhanced': enhanced_similarities}, output_file)

def calc_sim_euclidean(train_pairs: DatasetT, test_pairs: DatasetT, embeddings: Dict[int, np.ndarray]) -> DatasetT:
    pairs = train_pairs + test_pairs
    return [datautils.SentencePairT(s1=p.s1, s2=p.s2, score=1/(1+np.linalg.norm(embeddings[p.s1]-embeddings[p.s2]))) for p in pairs]

def calc_sim_regression(train_pairs: DatasetT, test_pairs: DatasetT, embeddings: Dict[int, np.ndarray]) -> DatasetT:
    def prepare_dataset(pairs):
        return [(np.concatenate((embeddings[p.s1], embeddings[p.s2])), p.score) for p in pairs]
    def normalize(data):
        return torch.nn.functional.normalize(data)
    dataloader = torch.utils.data.DataLoader(prepare_dataset(train_pairs), batch_size=16)
    model = SimilarityModel(len(embeddings[train_pairs[0].s1])*2)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bar = tqdm(range(1000), desc="Epoch")
    for epoch in bar:
        for inputs, targets in dataloader:
            # inputs = normalize(inputs.float())
            inputs = inputs.float()
            targets = targets.reshape((-1, 1)).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            bar.set_description(f"[{loss.item():.6f}] Epoch")
    test_inputs, test_targets = [torch.tensor(np.array(x)).float() for x in zip(*prepare_dataset(test_pairs))]
    test_targets = test_targets.reshape((-1, 1))
    test_outputs = model(test_inputs)
    test_loss = loss_function(test_outputs, test_targets)
    print("Test set loss:", test_loss.item())
    return [datautils.SentencePairT(s1=p.s1, s2=p.s2, score=s) for p,s in zip(test_pairs, test_outputs.squeeze().tolist())]

def calc_similarity_all(train_pairs: DatasetT, test_pairs: DatasetT, embeddings: Dict[int, np.ndarray]) -> DatasetT:
    return calc_sim_regression(train_pairs, test_pairs, embeddings)

def filter_pairs(dataset: DatasetT, valid_sentences: List[int]) -> DatasetT:
    return [p for p in dataset if p.s1 in valid_sentences and p.s2 in valid_sentences]

def main():
    torch.manual_seed(41)
    print("Loading everything...")
    dataset = datautils.load_dataset_mapping()
    graph_embs = load_graph_embeddings()
    sent_embs = load_sentence_embeddings()

    enhanced_embs: Dict[int, np.ndarray] = {}
    for sid in dataset.mapping:
        if sid in sent_embs and sid in graph_embs:
            sent_emb = sent_embs[sid]
            add_emb = graph_embs[sid]
            enhanced_embs[sid] = np.concatenate((sent_emb, add_emb))

    train_valid_dataset = filter_pairs(get_by_indices(dataset.dataset, dataset.split.train), list(enhanced_embs.keys()))
    test_valid_dataset = filter_pairs(get_by_indices(dataset.dataset, dataset.split.test), list(enhanced_embs.keys()))
    print(f"There are {len(train_valid_dataset)} valid train pairs and {len(test_valid_dataset)} valid test pairs")

    sent_similarities = calc_similarity_all(train_valid_dataset, test_valid_dataset, sent_embs)
    graph_similarities = calc_similarity_all(train_valid_dataset, test_valid_dataset, graph_embs)
    enhanced_similarities = calc_similarity_all(train_valid_dataset, test_valid_dataset, enhanced_embs)

    save_output(sent_similarities, graph_similarities, enhanced_similarities)

if __name__ == "__main__":
    main()
