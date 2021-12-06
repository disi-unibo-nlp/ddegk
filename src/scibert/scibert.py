# -*- coding: utf-8 -*-
from transformers import AutoModel, AutoTokenizer
from itertools import starmap
import torch
from typing import List, Tuple

CHUNKS_OVERLAP = 200

_scibert_tokenizer = None
_scibert = None
def _load_model():
    global _scibert_tokenizer
    global _scibert
    print('Loading SciBERT model...')
    _scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    _scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

def _span_by_tok(span_by_char: Tuple[int, int], mapping: List[Tuple[int, int]]) -> Tuple[int, int]:
    # Return the span by-token given the span by-character and the mapping obtained from the tokenizer.
    start, end = None, None
    for i, off in enumerate(mapping):
        if off[1] == 0:
            continue
        if start is None and off[0] <= span_by_char[0] < off[1]:
            start = i
        if end is None and off[0] < span_by_char[1] <= off[1]: # The first occurrence is saved
            end = i+1
    return start, end

def _tokenize(texts: List[str], max_len=512):
    # If a text is too long, it is truncated in chunks of the max size, as if they where different texts.
    # The first CHUNKS_OVERLAP tokens of every chunk are the same of the last CHUNKS_OVERLAP tokens of the previous one (not considering special tokens eg CLS).
    # Every element in overflow_mappings is the index of the original text it comes from. Indices in offset mappings refers to the original unchunked text.
    input_data = _scibert_tokenizer(texts, padding=True, return_offsets_mapping=True, truncation=True, max_length=max_len, stride=CHUNKS_OVERLAP, return_overflowing_tokens=True, return_tensors='pt')
    offset_mappings = input_data['offset_mapping']
    del input_data['offset_mapping']
    overflow_mappings = input_data['overflow_to_sample_mapping']
    del input_data['overflow_to_sample_mapping']
    return input_data, offset_mappings, overflow_mappings

def _rebuild_overflowed_tokens(overflow_mappings, data):
    # Returns a list of tensors of elements from a 2D matrix where multiple lines can belong to the same original (overflowed) tokens list,
    # given also the overflow_mappings from the tokenizer.
    overflow_ends = torch.cumsum(torch.unique(overflow_mappings, return_counts=True)[1], dim=0)
    overflow_starts = torch.roll(overflow_ends, 1)
    overflow_starts[0] = 0
    data = data[:, 1:-1] # Ignore special tokens
    # The start and end indeces of valid data to keep and leave out the stride
    valid_start = CHUNKS_OVERLAP // 2
    valid_end = len(data[0]) - (CHUNKS_OVERLAP - valid_start)
    # Every iteration gets the span of rows belonging to the same original text
    grouped_offset_mappings = []
    for group_span in zip(overflow_starts, overflow_ends):
        if group_span[1] == group_span[0] + 1:
            new_list = data[group_span[0]]
        else:
            new_list = torch.cat([
                data[group_span[0], :valid_end], # Don't need to remove anything from the start of the first row
                data[group_span[0]+1:group_span[1]-1, valid_start:valid_end].flatten(end_dim=1),
                data[group_span[1]-1, valid_start:] # Don't need to remove anything from the end of the last row
            ], dim=0)
        grouped_offset_mappings.append(new_list)
    return grouped_offset_mappings
    
def _get_span_embeddings_batch(texts: List[str], spans: List[List[Tuple[int, int]]]) -> List[List[List[float]]]:
    input_data, offset_mappings, overflow_mappings = _tokenize(texts)
    embeddings = _scibert(**input_data)['last_hidden_state']

    flat_embeddings = _rebuild_overflowed_tokens(overflow_mappings, embeddings)
    flat_offset_mappings = _rebuild_overflowed_tokens(overflow_mappings, offset_mappings)

    spans_by_tok = [[_span_by_tok(s, m) for s in ss] for m, ss in zip(flat_offset_mappings, spans)]

    return [[torch.cat((emb[s[0]], torch.mean(emb[s[0]:s[1]], dim=0), emb[s[1]-1])).tolist() for s in ss] for emb, ss in zip(flat_embeddings, spans_by_tok)]


def gen_span_embeddings(texts: List[str], spans: List[List[Tuple[int, int]]], batch_size: int) -> List[List[List[float]]]:
    '''A generator yielding the embeddings calculated with SciBERT for the given spans of the given texts.
    Spans must be pairs of character indices: if they don't align with the SciBERT tokens, the span will be expanded to include whole tokens.
    Providing multiple texts/spans lists to calculate in batch.
    The result is a torch.Tensor for each span, concatenation of:
     [emb of first token, mean embedding of the span, emb of last token]
    '''
    if _scibert is None:
        _load_model()

    if len(texts) != len(spans):
        raise Exception('Length of the texts and span lists must be the same')
    if batch_size <= 0:
        raise Exception('The batch size must be positive')

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        embeddings = _get_span_embeddings_batch(texts[start:end], spans[start:end])
        for e in embeddings:
            yield e
