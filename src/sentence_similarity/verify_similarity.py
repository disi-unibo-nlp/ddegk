from typing import Dict, Tuple, List
import datautils
import json
import numpy as np
from utils import map_keys
from functools import reduce

CALC_DATASET_FILENAME = "similarities.json"

def load_calculated_dataset() -> Dict[str, datautils.DatasetT]:
    def parse_list(l):
        return [datautils.SentencePairT(*p) for p in l]
    with open(CALC_DATASET_FILENAME) as input_file:
        data = json.load(input_file)
    return map_keys(parse_list, data)

def scores_mapping(dataset: datautils.DatasetT) -> Dict[int, float]:
    def key(p: datautils.SentencePairT):
        return p.s1*100000 + p.s2
    return {key(p): p.score for p in dataset}

def calc_score(real, calculated):
    print("Scoring with MSE")
    return np.square(real - calculated).mean()

def main():
    real_scores_map = scores_mapping(datautils.load_dataset_mapping().dataset)
    calculated_datasets = load_calculated_dataset()
    calc_scores_maps = map_keys(scores_mapping, calculated_datasets)
    valid_keys: List[int] = sorted(reduce(set.intersection, map(set, map(dict.keys, calc_scores_maps.values()))))
    real_values = np.array([real_scores_map[k] for k in valid_keys])
    calc_values = map_keys(lambda x: np.array([x[k] for k in valid_keys]), calc_scores_maps)
    print(f"{len(valid_keys)} valid pairs")
    for name, values in calc_values.items():
        print(f"Score for dataset '{name}' is {calc_score(real_values, values)}")


if __name__ == "__main__":
    main()
