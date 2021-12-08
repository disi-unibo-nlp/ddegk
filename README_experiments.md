# DDEGK Experiments

## Installing dependencies
Dependencies are listed in the file `requirements.txt`, they can be installed with `pip install -r requirements.txt`.  
**It is necessary to use Python 3.6**.

## Events to graph embeddings
1. _Prepare the dataset_. Create a folder under `data` named as your dataset. It must contain a list of [BioNLP Standoff](http://2011.bionlp-st.org/home/file-formats) files, i.e., triples composed of a text document (`.txt`), entity (`.a1`), and event (`.a2`) annotations. It is also possible to provide `.ann` files for the annotations instead.
2. _Run DDEGK_. Execute `python -m src.events_embedding --help` to see all the available parameters.
3. _Analyze the results_. The output will be saved in the folder `results/<datasetname>/ddegk/results.json`. The provided Jupyter notebooks can be used to visualize the results.

An example command:

```
python -m src.events_embedding --dataset=test --node-embedding-coeff=1 --node-label-coeff=1 --edge-label-coeff=1 --prototype-choice=random --num-prototypes=2
```
