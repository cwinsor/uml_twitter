'''
Dynamic topic modeling of GeoCoV19 using self-supervised graph learning

Step 1: preprocessing

This file preprocesses the rehydrated GeoCoV19 .ijson files into
PyTorch Geometric HeteroData (torch_geometric.data.HeteroData).
Such can then be read using DataLoader.

Topic embeddings are introduced <ZONA details here>

References:
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html#
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
https://www.sbert.net/ (sentence transformer-based encoding)
'''

import argparse
import logging

from geocov19_graph_dataset import GeoCoV19GraphDataset

# runtime arguments
parser = argparse.ArgumentParser(description="Pytorch Geometric Twitter Dataset Preprocessing")

parser.add_argument("--root_dir", type=str, required=True, help="data files folder")


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    # constructing an instance will perform the conversion including saving .pt files
    dataset = GeoCoV19GraphDataset(root=args.root_dir)

    logger.info("------ data[0] ------")
    data = dataset[0]
    logger.info(data)

    logger.info("------ data[1] ------")
    data = dataset[0]
    logger.info(data)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
