from typing import Callable, Dict, TypeVar, List

T = TypeVar("T")
V = TypeVar("V")
R = TypeVar("R")
def map_keys(fn: Callable[[V], R], d: Dict[T, V]) -> Dict[T, R]:
    return {k: fn(v) for k,v in d.items()}

def get_by_indices(l: List[T], indeces: List[int]) -> List[T]:
    return [l[i] for i in indeces]
