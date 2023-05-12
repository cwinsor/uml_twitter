import argparse
import json

import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import torch.nn.functional as F

'''
Heterogeneous Graph Train/Test

References:

Sample code:
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/load_csv.py
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/*

Tutorial:
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html

Non-torch GNN
https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3

Self-supervised training on Graphs, and top-level view:
https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809

'''

# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument("--src_folder", type=str, required=True)


def load_nodes_from_file(filename, encoders=None, **kwargs):

    with open(args.src_folder + filename, "r", encoding="utf-8") as f:
        the_data = json.load(f)

    mapping = {index: i for i, index in enumerate(the_data.keys())}

    x = torch.Tensor()
    if encoders is not None:
        for col, encoder in encoders.items():
            column_of_data = [v[col] for _, v in the_data.items()]
            xs = encoder(column_of_data)
            x = torch.cat((x, xs), dim=-1)

    return x, mapping


def load_edges_from_file(filename, src_index_col, src_mapping, dst_index_col, dst_mapping,
                         encoders=None, **kwargs):

    with open(args.src_folder + "\\" + filename, "r", encoding="utf-8") as f:
        the_data = json.load(f)

    src = [src_mapping[row[src_index_col]] for row in the_data]
    dst = [dst_mapping[row[dst_index_col]] for row in the_data]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    # if encoders is not None:
    #     edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
    #     edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class SequenceEncoder:
    r"""'SequenceEncoder' encodes raw column strings into embeddings."""
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, sequence):
        print(f"sequence encoding using {self.model}")
        x = self.model.encode(sequence, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class DateEncoder:
    r"""DateEncoder encodes month/day columns into embeddings."""
    def __init__(self, device=None):
        self.device = device

    @torch.no_grad()
    def __call__(self, ts):

        monthmap = {
            "Jan": 0,
            "Feb": 1,
            "Mar": 2,
            "Apr": 3,
            "May": 4,
            "Jun": 5,
            "Jul": 6,
            "Aug": 7,
            "Sep": 8,
            "Oct": 9,
            "Nov": 10,
            "Dec": 11,
        }

        source = [row.split(' ') for row in ts]
        embeddings = [[monthmap[row[1]], int(row[2])] for row in source]
        x = torch.Tensor(embeddings)
        return x.cpu()


class IdentityEncoder:
    r"""IdentityEncoder' takes the raw column values and converts them to PyTorch tensors."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class GeoCov19HeteroGNN(torch.nn.Module):
    r"""GeoCov19HeteroGNN is a heterogeneous graph based on data from the GeoCov19 Dataset.
    We are following https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#using-the-heterogeneous-convolution-wrapper"""
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('retweet', 'of', 'original_tweet'): GCNConv(-1, hidden_channels, add_self_loops=False),
                ('original_tweet', 'rev_of', 'retweet'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['author'])


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("create embeddings for original tweets using BERT-base-uncased")
    original_tweet_x, original_tweet_mapping = load_nodes_from_file(
        filename="originals_filtered.jsonl",
        encoders={
                'text': SequenceEncoder('bert-base-uncased'),
            })
    print(f"original tweets nodes and features: {original_tweet_x.shape}")

    retweet_x, retweet_mapping = load_nodes_from_file(
        filename="re_tweets_filtered.jsonl",
        encoders={
                'date': DateEncoder(),
            })
    print(f"retweet nodes and features: {retweet_x.shape}")

    edge_index, edge_label = load_edges_from_file(
        "list_o_r_filtered.jsonl",
        src_index_col=0,
        src_mapping=original_tweet_mapping,
        dst_index_col=1,
        dst_mapping=retweet_mapping,
        encoders=None,
    )

    data = HeteroData()
    data['original_tweet'].x = original_tweet_x
    data['retweet'].x = retweet_x
    data['retweet', 'of', 'original_tweet'].edge_index = edge_index

    # We can now convert `data` into an appropriate format for training a
    # graph-based machine learning model:

    # 1. Add a reverse relation for message passing.
    data = ToUndirected()(data)
    del data['original_tweet', 'rev_of', 'retweet'].edge_label  # Remove "reverse" label.

    # 2. Perform a link-level split into training, validation, and test edges.
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('retweet', 'of', 'original_tweet')],
        rev_edge_types=[('original_tweet', 'rev_of', 'retweet')],
    )
    train_data, val_data, test_data = transform(data)
    print(f"---train_data---\n{train_data}")
    print(f"---val_data---\n{val_data}")
    print(f"---test_data---\n{test_data}")

    # zona = data.num_classes
    model = GeoCov19HeteroGNN(hidden_channels=64,
                              # out_channels=data.num_classes,
                              out_channels=3,  # ZONA
                              num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    def train():
        model.train()
        optimizer.zero_grad()

        # d(self, x_dict, edge_index_dict):

        out = model(x_dict=train_data.x_dict,
                    edge_index_dict=train_data.edge_index_dict)
        loss = F.mse_loss(out, train_data['retweet', 'original_tweet'].edge_label)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(data):
        model.eval()
        out = model(
            data.x_dict,
            data.edge_index_dict,
            data['user', 'movie'].edge_label_index,
        ).clamp(min=0, max=5)
        rmse = F.mse_loss(out, data['user', 'movie'].edge_label).sqrt()
        return float(rmse)

    for epoch in range(1, 3):
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
            f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
