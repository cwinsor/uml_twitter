'''
Geo19Data Class

Derived from torch_geometric.data Data
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs

In the context of GeoCoV19 a data element is a graph consisting of
an original tweet and history of retweets. Tweet = node, retweets=nodes.
Each retweet is associated with the original tweet by an edge.

------------------------------------------
The following is background on the base class torch_geometric.Data:

The torch_geometric Data class describes a homogeneous graph.
An object of this class consists of nodes, edges.
    x is the node feature matrix [num_nodes, num_node_features]
    edge_index is the connectivity in COO format [2, num_edges]
    y is the ground truth label
Optional are:
    edge_attr
    pos (node position matrix)

Above describes a homogeneous graph.
The to_hetrogenous method can be used to convert the graph to a hetrogeneous format.
The following parameters are used to perform the conversion:
to_heterogeneous(node_type, edge_type, node_type_names, edge_type_names)
    node_type: a vector denoting node type
    edge_type: a vector indicating edge type
node_type_names and edge_type_names can be added to be more helpful.

'''

import os
import argparse
import ijson

import torch
from torch_geometric.data import Data

# references:
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data

# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_test", default=False, action="store_true")


class G19Data(Data):

    def __init__(self, x, edge_index, y):
        super().__init__(x=x, edge_index=edge_index, y=y)


def test(args):

    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(f"data {data}")
    print(f"data.num_nodes {data.num_nodes}")
    print(f"data.num_edges {data.num_edges}")
    print(f"data.num_node_features {data.num_node_features}")
    print(f"data.has_isolated_nodes() {data.has_isolated_nodes()}")
    print(f"data.has_self_loops() {data.has_self_loops()}")
    print(f"data.is_directed() {data.is_directed()}")
    print()

    node_types = torch.tensor([[0], [1], [0]], dtype=torch.long)
    # node_type_names = ["seven", "eight"]
    # data.to_heterogeneous(node_type=node_types, node_type_names=node_type_names)
    data.to_heterogeneous(node_type=node_types)

    print(f" data.num_node_types {data.num_node_types }")
    print(f" data.num_edge_types {data.num_edge_types  }")
    print(f" data.get_all_tensor_attrs() {data.get_all_tensor_attrs()  }")
    print(f" data.has_isolated_nodes() {data.has_isolated_nodes()  }")
    print(f" data.has_self_loops() {data.has_self_loops()  }")
    print(f" data.is_coalesced() {data.is_coalesced()  }")
    print(f" data.is_directed() {data.is_directed()  }")
    print(f" data.is_undirected() {data.is_undirected()  }")
    print(f" data.node_attrs() {data.node_attrs()  }")
    print(f" data.num_edges {data.num_edges  }")
    print(f" data.num_nodes {data.num_nodes  }")
    print(f" data.put_edge_index() {data.put_edge_index()  }")
    print(f" data.remove_edge_index() {data.remove_edge_index()  }")
    print(f" data.size {data.size  }")
    print(f" data.get_all_edge_attrs() {data.get_all_edge_attrs()  }")

                 

if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_test:
        test(args)
