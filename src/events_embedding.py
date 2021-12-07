import argparse
import os
import shutil

# We must use PyTorch because DDGK needs an old version of TF
os.environ['USE_TORCH'] = 'TRUE'

import src.ddgk.main as ddgk
import src.scibert.main as scibert
import src.scibert.standoff2graphs as standoff2graphs

def parse_args():
    parser = argparse.ArgumentParser()
    ddgk.add_hypeparameters_arguments(parser)
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to work with")
    parser.add_argument("--scibert-batch-size", type=int, default=2)
    return parser.parse_args()

def main(args):
    dataset_name = args.dataset
    output_dir = f"results/{dataset_name}"
    args.dataset = f"data/{dataset_name}"
    args.output_file = os.path.join(output_dir, 'graphs.json')
    args.batchsize = args.scibert_batch_size
    args.graphs_file = args.output_file
    args.working_dir = os.path.join(output_dir, 'ddegk')
    print(args)

    if os.path.isdir(args.working_dir):
        shutil.rmtree(args.working_dir)
    os.makedirs(args.working_dir)


    print("==== Generate SciBERT entity embeddings ====")
    scibert.main(args)
    print("\n==== Convert events to graphs ====")
    standoff2graphs.main(args)
    print("\n ==== Calculate graph embeddings ====")
    ddgk.main(args)


if __name__ == "__main__":
    main(parse_args())
