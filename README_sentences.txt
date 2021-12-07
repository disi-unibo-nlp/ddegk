# Event based sentence similarity

**Goal**: improve sentence similarity by using events.

## Run it
1. Run `prepare_sents.py` 
2. Move folder `sentences` to `path/to/DeepEvenMine/data/`
3. Run `infer_all.sh` in DEM folder
4. Move `sentences_results/` to `scibert_embeddings`
5. Run `for model in sentences_results; do main.py --dataset $model/ev-last/sentences_brat; done`
6. Run `for model in sentences_results; do standoff2graphs.py --dataset $model/ev-last/sentences_brat; done`
