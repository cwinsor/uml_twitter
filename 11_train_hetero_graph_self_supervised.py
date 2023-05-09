
'''
Dynamic topic modeling of GeoCoV19 using self-supervised graph learning

This file trains the graph-based model using self-supervised learning.

References:
https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html

'''

import datetime
import argparse
import logging
import wandb
from tqdm import trange

import torch
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader, NeighborLoader, HGTLoader

from g19_hetero_dataset import GeoCoV19GraphDataset
from g19_hetero_data import G19HeteroData
from geocov19_model import GeoCoV19Model

# from GraphSSL.data import load_dataset, split_dataset, build_loader
# from GraphSSL.model import Encoder
from loss import infonce, jensen_shannon

# runtime arguments
parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str,
                    default=datetime.datetime.now().strftime("%m%d_%H%M%S"),
                    help="unique name for this run")
parser.add_argument("--data_src", type=str, required=True,
                    help="source data folder")
parser.add_argument("--data_dst", type=str, required=True,
                    help="destination data/logs folder, also used as wandb project name")

parser.add_argument("--device", type=str,
                    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers_dataloader", default=8, type=int)


parser.add_argument("--dataset", dest="dataset", action="store", required=True, type=str,
                    choices=["GeoCoV19", "proteins", "enzymes", "collab", "reddit_binary", "reddit_multi", "imdb_binary",
                             "imdb_multi", "dd", "mutag", "nci1"],
                    help="dataset on which you want to train the model")
parser.add_argument("--model", dest="model", action="store", default="gcn", type=str,
                    choices=["gcn", "gin", "resgcn", "gat", "graphsage", "sgc"],
                    help="he model architecture of the GNN Encoder")
parser.add_argument("--feat_dim", dest="feat_dim", action="store", default=16, type=int,
                    help="dimension of node features in GNN")
parser.add_argument("--layers", dest="layers", action="store", default=1, type=int,
                    help=" number of layers of GNN Encoder")
parser.add_argument("--loss", dest="loss", action="store", default="infonce", type=str,
                    choices=["infonce", "jensen_shannon"],
                    help="loss function for contrastive training")
parser.add_argument("--augment_list", dest="augment_list", nargs="*",
                    default=["edge_perturbation", "node_dropping"], type=str,
                    choices=["edge_perturbation", "diffusion", "diffusion_with_sample", "node_dropping",
                             "random_walk_subgraph", "node_attr_mask"],
                    help="augmentations to be applied as space separated strings")
parser.add_argument("--train_data_percent", dest="train_data_percent", action="store", default=1.0, type=float)


def run_batch(args, epoch, mode, dataloader, model, optimizer):
    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()
    else:
        assert False, "Wrong Mode:{} for Run".format(mode)

    losses = []
    contrastive_fn = eval(args.loss + "()")

    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            data.to(args.device)

            # readout_anchor is the embedding of the original datapoint x on passing through the model
            # readout_anchor = model((data.x_anchor, data.edge_index_anchor, data.x_anchor_batch))
            out = model(data.x_dict, data.edge_index_dict)

            # readout_positive is the embedding of the positively augmented x on passing through the model
            readout_positive = model((data.x_pos, data.edge_index_pos, data.x_pos_batch))

            # negative samples for calculating the contrastive loss is computed in contrastive_fn
            loss = contrastive_fn(readout_anchor, readout_positive)

            if mode == "train":
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # keep track of loss values
            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()

    # gather the results for the epoch
    epoch_loss = sum(losses) / len(losses)
    return epoch_loss


def checkpoint_filename(step, args):
    file_name = f"{args.output_dir}/{args.run_name}_{step}.pt"
    return file_name


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project=args.data_dst, config=args)

    # dataset
    full_dataset = GeoCoV19GraphDataset(root=args.data_src)
    num_classes = 2  # should be full_dataset.num_classes... see https://github.com/pyg-team/pytorch_geometric/issues/1323

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # dataloaders
    data_for_init = full_dataset[0]
    train_loader = NeighborLoader(data=data_for_init,
                                  num_neighbors=[-1],
                                  input_nodes='original_tweet')
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                           shuffle=True, num_workers=args.num_workers_dataloader)
    # val_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                         shuffle=True, num_workers=args.num_workers_dataloader)
    # test_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers_dataloader)

    # model
    model = GeoCoV19Model(hidden_channels=args.feat_dim,
                                  out_channels=num_classes,
                                  num_layers=args.layers)
    # lazy init - reference https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#using-the-heterogeneous-convolution-wrapper
    with torch.no_grad():  # Initialize lazy modules.
        # out = model(data_for_init.x_dict, data_for_init.edge_index_dict)
        out = model(data_for_init, data_for_init)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_train_loss, best_val_loss = float("inf"), float("inf")

    for epoch in range(args.epochs):

        train_loss = run_batch(args, epoch, "train", train_loader, model, optimizer)
        logger.info(f"epoch {epoch} train loss {train_loss}")

        val_loss = run_batch(args, epoch, "val", val_loader, model, optimizer)
        logger.info(f"epoch {epoch}   val loss {train_loss}")

        # save model
        if val_loss < best_val_loss:
            best_epoch, best_train_loss, best_val_loss, is_best_loss = epoch, train_loss, val_loss, True

            model.save_checkpoint(f"output_{args.projectname}",
                                  optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                step=epoch * args.batch_size
            )

    logger.info(f"Best validation loss at epoch {epoch}: val {best_val_loss:.3f} train {best_train_loss:.3f}")
    test_loss = run_batch(args, best_epoch, "test", test_loader, model, optimizer)
    logger.info(f"Test loss using model from epoch {best_epoch}: {test_loss:.3f}")
    wandb.log(
        {
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "test_loss (final)": test_loss,
        })


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
