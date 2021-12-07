# coding: utf-8

from typing import List, Set, Tuple, Dict, Union
import os
from glob import glob
import itertools

class DataFile:
    def __init__(self, root, filename):
        self.path = os.path.join(root, filename)
        self.name = filename
    @property
    def ann(self):
        return self.path + '.ann'
    @property
    def txt(self):
        return self.path + '.txt'

    def __repr__(self):
        return self.path

def data_files(dataset_root: str) -> List[DataFile]:
    def base_root_name(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]
    def find_by_ext(ext: str) -> Set[str]:
        return set(map(base_root_name, glob(os.path.join(dataset_root, f'*.{ext}'))))

    ann_files = find_by_ext('ann')
    txt_files = find_by_ext('txt')
    intersection = ann_files | txt_files
    union = ann_files & txt_files
    diff = intersection - union
    if len(diff) > 0:
        print("WARNING: Some ANN or TXT files are missing.\n" + ' '.join(diff))
    return [DataFile(dataset_root, f) for f in union]


class StandoffEntity:
    def __init__(self, _id: str, _type: str, _span: Tuple[int, int], _name: str):
        self.id = _id
        self.type = _type
        self.span = _span
        self.name = _name

    @staticmethod
    def from_line(line: str) -> 'StandoffEntity':
        assert line[0] == 'T'
        _id, _args, _name = line.split('\t')
        _type, _span_start, _span_end = _args.split(' ')
        return StandoffEntity(_id, _type, (int(_span_start), int(_span_end)), _name)

class StandoffEvent:
    def __init__(self, _id: str, _trigger: StandoffEntity, _arguments: List[Tuple[Union[StandoffEntity, 'StandoffEvent'], str]]):
        self.id = _id
        self.trigger = _trigger
        self.arguments = _arguments

    @staticmethod
    def from_line(line: str, entities: Dict[str, StandoffEntity], events: Dict[str, 'StandoffEvent']):
        _id, _others = line.split('\t')
        [_, _trigger], *_arguments = [a.split(':') for a in _others.split()]
        resolved_args = [(entities[a[1]] if a[1] in entities else events[a[1]], a[0]) for a in _arguments]
        return StandoffEvent(_id, entities[_trigger], resolved_args)


def load_document(doc: DataFile) -> Tuple[Dict[str, StandoffEntity], Dict[str, StandoffEvent], str]:
    with open(doc.txt) as txt_file:
        text = txt_file.read()
    
    with open(doc.ann) as ann_file:
        annotations = list(ann_file)

    entities = {}
    events = {}
    for line in annotations:
        if line[0] == 'T':
            ent = StandoffEntity.from_line(line)
            entities[ent.id] = ent
    
    repeat = True
    while repeat == True:
        repeat = False
        for line in annotations:
            if line[0] == 'E':
                try:
                    event = StandoffEvent.from_line(line, entities, events)
                    if event.id not in events:
                        events[event.id] = event
                except Exception as e:
                    repeat = True

    return entities, events, text
