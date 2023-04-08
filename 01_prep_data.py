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

import os
import glob
import argparse
import logging

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from geocov19_graph_dataset import GeoCoV19GraphDataset

# runtime arguments
parser = argparse.ArgumentParser(description="Pytorch Geometric Twitter Dataset Preprocessing")

parser.add_argument("--root_dir", type=str, required=True,
                    help="data files folder")

parser.add_argument("--force_rebuild",
                    default=False, action="store_true",
                    help="force rebuild of model files even if they already exist")

parser.add_argument("--revalidate_all_date_by_reading_from_file",
                    default=False, action="store_true",
                    help="revalidate model integrity by reading in as model from file")


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    if args.force_rebuild:
        filelist = glob.glob(args.root_dir + r"\processed\*")
        [os.remove(filename) for filename in filelist]

    # constructing an instance will perform the conversion including saving .pt files
    dataset = GeoCoV19GraphDataset(root=args.root_dir)

    # logger.info("------ data[0] ------")
    # data = dataset[0]
    # logger.info(data)

    # logger.info(f"data.num_node_features {data.num_node_features}")
    # logger.info(f"data.num_edge_features {data.num_edge_features}")
    # logger.info(f"data.num_features {data.num_features}")
    # logger.info(f"data.has_isolated_nodes() {data.has_isolated_nodes()}")
    # logger.info(f"data.is_undirected() {data.is_undirected()}")
    # logger.info(f"data.metadata() {data.metadata()}")
    # logger.info(f"data.node_items()[0]\n{data.node_items()[0]}")
    # logger.info(f"data.edge_items()[0]\n{data.edge_items()[0]}")

    if (args.revalidate_all_date_by_reading_from_file):
        dataset = GeoCoV19GraphDataset(root=args.root_dir)
        dataloader = DataLoader(dataset=dataset)
        for data in tqdm(dataloader):
            print("------ revalidating ---------")
            data.validate()

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
