# coding: utf-8

from typing import List, Dict, Tuple

from tqdm import tqdm
import json
import networkx
import argparse

import src.scibert.datautils as datautils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.dataset + '_graphs.json'
    return args

def main(args):
    datafiles = datautils.data_files(args.dataset)

    all_graphs = []
    for datafile in tqdm(datafiles, desc='Reading files'):
        entities, events, text = datautils.load_document(datafile)

        with open(datafile.embeddings) as ff:
            embeddings = json.load(ff)

        # Create a graph with all the entities and events
        graph = networkx.DiGraph(source_doc=datafile.base_name, dataset=args.dataset)
        for ent in entities.values():
            graph.add_node(ent.id, type=ent.type, name=ent.name, embedding=embeddings[ent.id])
        for event in events.values():
            for argument, role in event.arguments:
                arg_id = argument.id if type(argument) is datautils.StandoffEntity else argument.trigger.id
                graph.add_edge(event.trigger.id, arg_id, key=role, event_id=event.id)

        # Find all the "root" events (not nested)
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
        for root in roots:
            root_event = networkx.induced_subgraph(graph, networkx.descendants(graph, root) | set([root])).copy()
            root_event.graph['root'] = root
            all_graphs.append(networkx.node_link_data(root_event))

    print(f'Saving {len(all_graphs)} graphs...')
    with open(args.output_file, 'w') as ff:
        json.dump(all_graphs, ff)

if __name__ == '__main__':
    main(parse_args())
