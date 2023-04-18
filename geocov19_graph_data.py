'''
GeoCoV19GraphData Class

Derived from torch_geometric.data Data
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data

In the context of GeoCoV19 a graph is the tweet and history of retweets.
Tweet = node, retweets=nodes. Each retweet is associated with the original tweet by a vertex.

------------------------------------------
The following is background info on torch_geometric.Data:

The torch_geometric Data class describes a homogeneous graph.
An object of this class consists of nodes, edges.
    x is the node feature matrix [num_nodes, num_node_features]
    edge_index is the connectivity in COO format [2, num_edges]
    y is the ground truth label
Optional are:
    edge_attr
    pos (node position matrix)

Above describes a homogeneous graph. For hetrogeneous support the to_hetrogenous method can be called
on the homeogenous graph with the following parameters:
to_heterogeneous(node_type, edge_type, node_type_names, edge_type_names)
    node_type: a vector denoting node type
    edge_type: a vector indicating edge type
node_type_names and edge_type_names can be added to be more helpful.

The infrastructure will ten 
In our case:



Derived from torch InMemoryDataset.
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data

A graph captures a tweet's retweet events.
Nodes are the original tweet and each retweet.
An edge associates the retweet with the original tweet.

Thus there are millions of graphs and each is relatively small (the total number of retweets + 1)

The goal is to classify a tweet (entire graph) into one of three classes:
* "0" = little or no retweet activity
* "1" = modest retweet activity
* "2" = significant retweet activity

'''

import os
import ijson

import torch
from torch_geometric.data import Dataset, InMemoryDataset, HeteroData


class GeoCoV19GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    # @property
    # def num_classes(self):
    #     return 3

    @property
    def raw_file_names(self):
        # return ['ids_geo_2020-02-01.jsonl']
        # return ['ids_geo_2020-02-01.jsonl', 'ids_geo_2020-02-02.jsonl']
        return 'merged_all.jsonl'

    @property
    def processed_file_names(self):
        # return ['ids_geo_2020-02-01.pt']
        # return ['ids_geo_2020-02-01.pt', 'ids_geo_2020-02-02.pt']
        return [
            'data_0.pt', 'data_1.pt',
            ]

    def download(self):
        rfn = self.raw_file_names
        assert False, (f'did not find raw data {self.root}\\raw\\{rfn} and download is not implemented')

    def process(self):
        print("zona - here in 'process'")
        data_list = [...]

# get data from file
        file_handle = open(args.analyze_target, "r", encoding="utf-8")     
        data = ijson.items(file_handle, "", multiple_values=True)
        # for now read it all in - later this could be changed to 
        # data_list = 
        # num_retweets_list = [original_tweet["number_retweets"] for original_tweet in data]






        if self.pre_filter is not None:
            assert False, "zona - this is where a pre-filter would go"

        if self.pre-transform is not None:
            assert False, "zona - this is where a pre-transform would go"

    def _is_retweet(self, tweet):
        is_retweet = "retweeted_status" in tweet.keys()
        return is_retweet

    def _load_node_users(self, retweets, encoders=None):

        twitter_user_ids = {tweet["user"]["id"] for tweet in retweets}
        mapping = {twitter_user_id: my_user_id for my_user_id, twitter_user_id in enumerate(twitter_user_ids)}

        x = None
        x = torch.rand(len(mapping), 3)  # zona temporary
        # if encoders is not None:
        #     xs = [encoder(retweets[col]) for col, encoder in encoders.items()]
        #     x = torch.cat(xs, dim=-1)
        return x, mapping

    def _load_node_original_tweets(self, retweets, encoders=None):

        twitter_retweet_ids = {tweet["retweeted_status"]["id"] for tweet in retweets}
        mapping = {twitter_retweet_id: my_retweet_id for my_retweet_id, twitter_retweet_id in enumerate(twitter_retweet_ids)}

        x = None
        x = torch.rand(len(mapping), 4)  # zona temporary
        # if encoders is not None:
        #     xs = [encoder(retweets[col]) for col, encoder in encoders.items()]
        #     x = torch.cat(xs, dim=-1)
        return x, mapping



    def _load_edge_user_retweets_original_tweet(self, retweets, src_mapping, dst_mapping, encoders=None):
        src = [src_mapping[tweet["user"]["id"]] for tweet in retweets]
        dst = [dst_mapping[tweet["retweeted_status"]["id"]] for tweet in retweets]

        edge_index = torch.tensor([src, dst])

        edge_attr = None
        # if encoders is not None:
        #     edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        #     edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr

    def process(self):

        file_idx = -1
        for raw_path in self.raw_paths:
            file_idx += 1
            print(f"reading {raw_path}")

            # Read data from file
            with open(raw_path, "r", encoding="utf-8") as f:
                tweets = ijson.items(f, "", multiple_values=True)

                # just retweets
                retweets = [
                    tweet for tweet in tweets if self._is_retweet(tweet)
                    ]

                # NODES
                users_x, users_mapping = self._load_node_users(retweets=retweets, encoders=None)
                original_tweet_x, original_tweet_mapping = self._load_node_original_tweets(retweets=retweets, encoders=None)
                # EDGES
                edge_index, edge_label = self._load_edge_user_retweets_original_tweet(
                    retweets=retweets,
                    src_mapping=users_mapping,
                    dst_mapping=original_tweet_mapping,
                    encoders=None)

                data = HeteroData()
                data['user'].x = users_x
                data['original_tweet'].x = original_tweet_x
                data['user', 'retweets', 'original_tweet'].edge_index = edge_index
                # print(data)

                torch.save(data, os.path.join(self.processed_dir, f'data_{file_idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
