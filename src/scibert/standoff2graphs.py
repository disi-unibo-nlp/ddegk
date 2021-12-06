# coding: utf-8

from tqdm import tqdm
import json
import networkx
import argparse
import datautils

from typing import List, Dict, Tuple

def base_root_name(filename: str) -> str:
    o, _ = os.path.splitext(os.path.basename(filename))
    return o
    
def events_folder(dataset: str) -> str:
    return f'experiments/{dataset}/results/ev-last/ev-tok-ann'

def entities_folder(dataset: str) -> str:
    return f'experiments/{dataset}/results/rel-last/rel-ann'
    

def a2_file_path(dataset: str, filename: str) -> str:
    return os.path.join(events_folder(dataset), filename+'.a2')

def emb_file_path(dataset: str, filename: str) -> str:
    return os.path.join(entities_folder(dataset), filename+'-EMB.json')

def preprocessed_text_path(dataset: str, filename: str) -> str:
    return f'data/{dataset}/processed-text/text/{filename}.txt'

def find_span_line(text: str, span_start: int, span_end: int) -> int:
    # Find the line containing the given span. If the spans is on multiple lines,
    # or it is outside the given text, returns None.
    if span_start >= len(text):
        return None
    def line_for(idx: int) -> int:
        return text.count('\n', 0, idx)
    start_line = line_for(span_start)
    end_line = line_for(span_end)
    if start_line != end_line:
        return None
    return start_line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
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
    with open(args.dataset + '_graphs.json', 'w') as ff:
        json.dump(all_graphs, ff)

if __name__ == '__main__':
    main()
