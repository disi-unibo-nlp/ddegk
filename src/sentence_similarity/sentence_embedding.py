from sentence_transformers import SentenceTransformer
import datautils
import json

# MODEL_NAME = "all-MiniLM-L12-v2"
MODEL_NAME = "all-mpnet-base-v2"
OUTPUT_FILENAME = "sentences_emb.json"

def main():
    print("Loading everything...")
    dataset = datautils.load_dataset_mapping()
    model = SentenceTransformer(MODEL_NAME)

    sids, texts = zip(*list(dataset.mapping.items()))
    embeddings = [t.tolist() for t in model.encode(texts, show_progress_bar=True)]
    print("Saving to", OUTPUT_FILENAME)
    with open(OUTPUT_FILENAME, "w") as output_file:
        json.dump(dict(zip(sids, embeddings)), output_file)


if __name__ == "__main__":
    main()
