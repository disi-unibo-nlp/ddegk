from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict, is_dataclass
import collections
import json
import csv

MAPPED_DATASET_NAME = 'mapped_dataset.json'

@dataclass(repr=False, eq=False)
class SentencePairT:
    s1: str
    s2: str
    score: float

@dataclass(repr=False, eq=False)
class SentenceIndexPairT:
    s1: int
    s2: int
    score: float

DatasetT = List[SentencePairT]

@dataclass(repr=False, eq=False)
class Split:
    train: List[int]
    test: List[int]

@dataclass(repr=False, eq=False)
class MappedDataset:
    dataset: List[SentenceIndexPairT]
    mapping: Dict[int, str]
    split: Split

def _has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))

# def _custom_json_encoder(obj) -> object:
#     if is_dataclass(obj):
#         return asdict(obj)
#     raise TypeError("Not serializable: " + type(obj).__name__)

def load_dataset_mapping() -> MappedDataset:
    with open(MAPPED_DATASET_NAME) as input_file:
        data = json.load(input_file)
    data['dataset'] = [SentencePairT(*p) for p in data['dataset']]
    data['mapping'] = {int(k): v for k,v in data['mapping'].items()} # JSON are always strings
    data['split'] = Split(**data['split'])
    dataset = MappedDataset(**data)
    return dataset

def save_dataset_mapping(dataset: DatasetT, mapping: Dict[int, str], split: Split):
    inverse_mapping = {s: i for i, s in mapping.items()}
    mapped_dataset = [SentenceIndexPairT(
            s1=inverse_mapping[pair.s1],
            s2=inverse_mapping[pair.s2],
            score=pair.score
        ) for pair in dataset]
    with open(MAPPED_DATASET_NAME, 'w') as output_file:
        json.dump(asdict(MappedDataset(dataset=mapped_dataset, mapping=mapping, split=split)), output_file)

def create_sentence_map(pairs: DatasetT) -> Dict[int, str]:
    sent_set = set()
    for pair in pairs:
        sent_set.update([pair.s1, pair.s2])
    return dict(enumerate(sorted(sent_set)))

def load_original_datasets() -> DatasetT:
    return [SentencePairT(fix_unicode(p.s1), fix_unicode(p.s2), p.score) for p in _load_biosses_original() + _load_ctr_original()]

def _load_biosses_original() -> DatasetT:
    # Scale the score in [0, 1]
    pairs = []
    for fn in ['train', 'test', 'dev']:
        with open(f'data/BIOSSES/{fn}.tsv', newline='') as input_file:
            pairs += [SentencePairT(row['sentence1'], row['sentence2'], float(row['score'])/4) for row in csv.DictReader(input_file, dialect='excel-tab')]
    return pairs

def _load_ctr_original() -> DatasetT:
    # Pick the median of the score and scale in [0, 1]
    with open('data/ctr.csv', newline='') as input_file:
        return [SentencePairT(row['TxtA'], row['TxtB'], float(sorted([row['1'], row['2'], row['3']])[1])/4) for row in csv.DictReader(input_file, dialect='excel')]

def fix_unicode(text: str) -> str:
    return text.replace('ﬁ', 'fi').replace('−', '-')
