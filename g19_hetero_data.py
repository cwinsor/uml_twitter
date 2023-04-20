'''
Geo19HeteroData

References:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData

'''

import random
import argparse
import ijson

import torch
from torch_geometric.data import HeteroData


# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_test_1", default=False, action="store_true")
parser.add_argument("--run_test_2", default=False, action="store_true")
parser.add_argument("--data_file", type=str, required=False, default="_not_provided")


class G19HeteroData(HeteroData):

    def __init__(self):
        super().__init__()

    def from_json_standard_format(self, ref, the_data):

        original_tweet_id = the_data["original_tweet_id"]
        retweet_count = the_data["number_retweets"]
        # original_tweet_text = the_data["original_tweet_text"]  ## ZONA needs to be an INT or FLOAT, not string
        original_tweet_text_features = [random.randint(0, 9) for _ in range(3)]

        retweet_user_ids = the_data["retweet_user_ids"]
        # retweet_dates = the_data["retweet_dates"]  ## ZONA needs to be an INT or FLOAT, not datetime
        retweet_date_feature = [random.randint(0, 9) for _ in range(len(retweet_user_ids))]

        # node: original tweet
        self["original_tweet"].x = torch.tensor([
            [original_tweet_id, retweet_count] + original_tweet_text_features
            ])

        # nodes: retweets
        self['retweet'].x = torch.tensor([[id, date] for id, date in zip(retweet_user_ids, retweet_date_feature)])

        # edges...
        # each points to the original tweet
        node_index_to = [0 for (i, val) in enumerate(self['retweet'].x)]
        # each starts from the retweet
        node_index_from = [i for (i, val) in enumerate(self['retweet'].x)]

        edge_index_retweet_to_original_tweet = torch.tensor([node_index_from, node_index_to], dtype=torch.long)
        # print(edge_index_retweet_to_original_tweet)
        self['retweet', 'of', 'original_tweet'].edge_index = edge_index_retweet_to_original_tweet
        # print(self['retweet', 'of', 'original_tweet'].edge_index)
        # print("here")

    def print_summary(self):
        print(f"data {self}")
        print(f"metadata() {self.metadata()}")

        print(f"node_types {self.node_types}")
        print(f"edge_types {self.edge_types}")

        print(f"num_nodes {self.num_nodes}")
        print(f"num_node_features {self.num_node_features}")
        print(f"num_edges {self.num_edges}")
        print(f"num_edge_features {self.num_edge_features}")

        print(f"has_isolated_nodes() {self.has_isolated_nodes()}")
        print(f"is_directed() {self.is_directed()}")
        print(f"is_undirected() {self.is_undirected()}")
        print(f"has_self_loops() {self.has_self_loops()}")
        print(f"to_dict()\n{self.to_dict()}")


def test_2(args):

    f = open(args.data_file, "r", encoding="utf-8")
    graphs_from_file = ijson.items(f, "", multiple_values=True)

    for graph_data_from_file in graphs_from_file:

        if graph_data_from_file["number_retweets"] == 10:  # ZONA - delete me
            g19_hetero_data = G19HeteroData()

            g19_hetero_data.from_json_standard_format(ref=g19_hetero_data,
                                                      the_data=graph_data_from_file)

            g19_hetero_data.print_summary()

            assert False, "hold up"


def test_1(args):

    data = G19HeteroData()

    num_authors = 2
    num_authors_features = 5
    data['author'].x = torch.randn(num_authors, num_authors_features)

    num_papers = 3
    num_paper_features = 4
    data['paper'].x = torch.randn(num_papers, num_paper_features)

    edge_index_author_paper = torch.tensor([
        [0, 0, 1, 0],
        [1, 2, 3, 3]], dtype=torch.long)
    print(edge_index_author_paper.shape)  # should be 2x4
    data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

    data.print_summary()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_test_1:
        test_1(args)

    if args.run_test_2:
        test_2(args)
