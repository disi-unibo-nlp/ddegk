import src.scibert.datautils as datautils
import src.scibert.scibert as scibert
import argparse
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batchsize', type=int, default=2)
    args = parser.parse_args()

    print('Dataset:   ', args.dataset)
    print('Batch size:', args.batchsize)

    datafiles = datautils.data_files(args.dataset)
    if not datafiles:
        print('Empty dataset')
        return

    all_texts = []
    all_spans = []
    all_ent_ids = []
    for f in tqdm(datafiles, desc='Loading dataset'):
        entities, _, text = datautils.load_document(f)
        entities_vals = entities.values()
        spans = [e.span for e in entities_vals]
        ent_ids = [e.id for e in entities_vals]
        all_texts.append(text)
        all_spans.append(spans)
        all_ent_ids.append(ent_ids)

    all_embeddings_gen = scibert.gen_span_embeddings(all_texts, all_spans, args.batchsize)
    
    for df, embeddings, ent_ids in tqdm(zip(datafiles, all_embeddings_gen, all_ent_ids), desc='Generating and saving', total=len(all_texts)):
        with open(df.embeddings, 'w') as output_file:
            data = {eid: emb for eid, emb in zip(ent_ids, embeddings)} 
            json.dump(data, output_file)

if __name__ == '__main__':
    main()
