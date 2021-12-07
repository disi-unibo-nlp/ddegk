from typing import Tuple
import os
import src.sentence_similarity.datautils as datautils
import random

OUT_FOLDER_NAME = 'sentences/text'

def split_dataset(dataset_size: int) -> datautils.Split:
    # Return 2 lists of indices (train, test)
    TRAIN_SIZE = 0.8
    indeces = list(range(dataset_size))
    random.shuffle(indeces)
    split_index = int(TRAIN_SIZE * len(indeces))
    return datautils.Split(train=indeces[:split_index], test=indeces[split_index:])

def main():
    random.seed(40)
    dataset = datautils.load_original_datasets()
    all_sentences = datautils.create_sentence_map(dataset)
    print(f'Loaded {len(all_sentences)} unique sentences')

    train_test = split_dataset(len(dataset))

    print('Saving preprocessed dataset')
    datautils.save_dataset_mapping(dataset, all_sentences, train_test)

    print('Creating DeepEvenMine dataset')
    os.makedirs(OUT_FOLDER_NAME, exist_ok=True)
    for i, sent in all_sentences.items():
        with open(os.path.join(OUT_FOLDER_NAME, f'sent{i}.txt'), 'w') as ff:
            ff.write(sent)

if __name__ == '__main__':
    main()
