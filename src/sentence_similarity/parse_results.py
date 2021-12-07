from typing import Dict, List, Tuple, Union
import standoff
from standoff import StandoffEvent, StandoffEntity, DataFile
import json
import os
import re
import datautils
import graphsutils
import networkx as nx
import itertools

FOLDER = 'sentences_results'
RESULTS_PATH = 'ev-last/sentences-brat_graphs.json'
OUTPUT_FILENAME = 'graphs.json'

def get_dem_models() -> List[str]:
    return [d.name for d in filter(os.DirEntry.is_dir, os.scandir(FOLDER))]

def load_results_graphs(dataset_name: str) -> List[nx.Graph]:
    return graphsutils.load_graphs(os.path.join(FOLDER, dataset_name, RESULTS_PATH))

def deduplicate_events(events: List[nx.Graph]) -> List[StandoffEvent]:
    def key(graph: nx.Graph) -> str:
        return '|'.join(sorted(map(str, graph.edges.data('key')))).lower()
    if len(events) <= 1:
        return events
    mapped_events = [nx.relabel_nodes(g, {n: re.sub(r'\s+', '', g.nodes[n]['name']) for n in g.nodes}) for g in events]
    for mapped, original in zip(mapped_events, events):
        mapped.graph['original'] = original
    events = mapped_events
    ev_set: Dict[str, List[nx.Graph]] = {}
    for e in events:
        ev_set.setdefault(key(e), []).append(e)
    return [l[0].graph['original'] for l in ev_set.values()]

def main():
    dem_models = get_dem_models()
    print('Found results for DEM models:', ', '.join(dem_models))

    dataset = datautils.load_dataset_mapping()

    events: Dict[int, List[nx.Graph]] = {} # Dict[sentID, List[events]]

    # Load DeepEventMine results
    for model in dem_models:
        result_graphs = load_results_graphs(model)
        print(f'Found {len(result_graphs)} graphs for {model}')
        for g in result_graphs:
            sent_id = int(g.graph['source_doc'][len('sent'):])
            events.setdefault(sent_id, []).append(g)
    print(f'{len(events)}/{len(dataset.mapping)} sentences have at least one event')
    print(f'{sum(1 for x in events.values() if len(x) > 1)}/{len(events)} sentences have multiple events')

    # Count valid pairs
    valid = 0
    for pair in dataset.dataset:
        if pair.s1 in events and pair.s2 in events:
            valid += 1
    print(f'{valid}/{len(dataset.dataset)} valid pairs')

    # Deduplicate events
    previous_total = sum(len(x) for x in events.values())
    for sid, ev_list in events.items():
        events[sid] = deduplicate_events(ev_list)
        # input()
    deduplicated_total = sum(len(x) for x in events.values())
    print(f'Deduplicated events from {previous_total} to {deduplicated_total}')
    print(f'{sum(1 for x in events.values() if len(x) > 1)}/{len(events)} sentences have multiple events after deduplication')
    graphsutils.save_graphs(itertools.chain.from_iterable(events.values()), graphsutils.DEFAULT_FILENAME)

if __name__ == '__main__':
    main()
