# DDEGK Experiments

## Installing dependencies
The dependencies list is in the file `requirements.txt`, they can be installed with `pip install -r requirements.txt`.  
**It is necessary to use Python 3.6**.

## Events to graph embeddings
1. Prepare the dataset: create a folder under `data` named as your dataset. It must contain a list of [Standoff](http://2011.bionlp-st.org/home/file-formats) documents: for each document a `.txt` file containing the textual data, and `.a1` and `.a2` files for the annotations. It is also possible to provide `.ann` files for the annotations instead.
2. Run DDEGK: execute `python -m src.events_embedding --help` to see all the available parameters.
3. Analyze the results: the output will be saved in the folder `results/<datasetname>/ddegk/results.json`. The provided Jupyter notebooks can be used to visualize the results.

An example command:

```
python -m src.events_embedding --dataset=test --node-embedding-coeff=1 --node-label-coeff=1 --edge-label-coeff=1 --prototype-choice=random --num-prototypes=2
```
