'''
G19HeteroDataset

References:
https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset
'''

import argparse
import ijson
import json

import torch
from torch_geometric.data import InMemoryDataset
from g19_hetero_data import G19HeteroData


# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_test_1", default=False, action="store_true")
parser.add_argument("--run_test_2", default=False, action="store_true")
parser.add_argument("--data_root", type=str, required=False, default="_not_provided")


class GeoCoV19GraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["zona_fix_me_to_not_skip_process.txt"]
   
    def download(self):
        pass

    # def _get_tweets_and_retweets():

    #     # make a map: original tweet to [list of retweets]
    #     filepath = "D:\\dataset_covid_GeoCovGrHomogeneous\\raw\\merged_original_tweets.json"
    #     print(f"processing {filepath}")
    #     original_tweets = {}
    #     with open(filepath, "r", encoding="utf-8") as f_in:
    #         original_tweets = json.load(f_in)
    #     f_in.close()

    #     # here we get the original tweets as a dict with an empty list to start
    #     original_to_retweet_list = {}
    #     for k, v in original_tweets.items():
    #         original_to_retweet_list[k] = []

    #     # get the list of retweets
    #     filepath = "D:\\dataset_covid_GeoCovGrHomogeneous\\raw\\merged_retweets.json"
    #     print(f"processing {filepath}")
    #     retweets = {}
    #     with open(filepath, "r", encoding="utf-8") as f_in:
    #         retweets = json.load(f_in)
    #     f_in.close()

    #     # iterate through retweets appending each to the original tweet's list
    #     for retweet_id, v in retweets.items():
    #         original_tweet_id = v['fk_original_tweet']
    #         original_to_retweet_list[original_tweet_id].append(retweet_id)

    #     # the majority of tweets only get 1 retweet which is not interesting so get a reasonable sample
    #     MIN_RETWEETS = 6
    #     MAX_RETWEETS = 8
    #     original_to_retweet_selected = {}
    #     for orig_tw, ret_list in original_to_retweet_list.items():
    #         if len(ret_list) >= MIN_RETWEETS and len(ret_list) <= MAX_RETWEETS:
    #             original_to_retweet_selected[orig_tw] = ret_list

    #     return original_to_retweet_selected, original_tweets, retweets
    
    def process(self):

        # nodes_type 1 = dictionary[original_tweet_id] -> dictionary["original_text", "other_features..."]
        # nodes_type_2 = dictionary[retweet_id] -> dictionary['date', 'fk_original_tweet_id', "other_features..."]
        # type2_to_type_2 = dictionary[original_tweet_id] -> list[retweet_ids]
        type1_to_type2, nodes_type_1, nodes_type_2 = self._get_tweets_and_retweets()

        # following
        # https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

        data_list = []

        for original_tweet_id, retweet_list in type1_to_type2.items():
            graph = _make_data_object(original_tweet_id, retweet_list, nodes_type_1, nodes_type_2)
            data_list.append(graph)

        def _make_data_object(original_tweet_id, retweet_list, nodes_type_1, nodes_type_2):
            # following https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#creating-heterogeneous-graphs

            data = HeteroData()

            features_original_tweet = torch.rand(4)  # (to be) Hugging Face BERT embeddings
            data['original_tweet'].x = features_original_tweet  # [1 original tweet, num_features_original_tweet]

            features_retweets = torch.rand((len(retweet_list), 3))  # (to be) retweet features such as yy/mm/dd
            data['retweets'].x = TorchTensor()  # [num_retweets, num_features_retweet]

            edge_list = [ [retweet_id, original_tweet_id] for retweet_id in retweet_list ]
            data['retweet', 'of', 'original_tweet']  # [2, num_edges_retweets]

            ot_data = [ot_v, ]
            ot_nodes.append

        filepath = self.root + "/raw/merged_original_tweets.json",
        print(f"processing {filepath}")
        with open(filepath, "r", encoding="utf-8") as myfile:
            data = json.load(myfile)
            # print(f"len data {len(data)}")
            pass


        # Read data into huge `Data` list.
        data_list = []
        f = open(self.root + "/raw/merged_original_tweets.json", "r", encoding="utf-8")
        f = open(self.root + "/raw/merged_retweets.json", "r", encoding="utf-8")
        graphs_from_file = ijson.items(f, "", multiple_values=True)

        for graph_data_from_file in graphs_from_file:

            # if (graph_data_from_file["number_retweets"] > 20) and \
            #    (graph_data_from_file["number_retweets"] < 30):  # ZONA delete me
            if True:

                g19_hetero_data = G19HeteroData()

                g19_hetero_data.from_json_standard_format(ref=g19_hetero_data,
                                                          the_data=graph_data_from_file)

                # g19_hetero_data.print_summary()
                data['original_tweet'].x = g19_hetero_data.x.to(torch.long)
                data['original_tweet'].y = g19_hetero_data.x.to(torch.long)
                for retweet in g19_hetero_data['retweet_list']:
                    data['retweet']


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def test_1(args):

    gcgd = GeoCoV19GraphDataset(root=args.data_root)

    # print(f"raw_file_names {gcgd.raw_file_names}")
    print(f"processed_file_names {gcgd.processed_file_names}")
    # print(f"raw_paths {gcgd.raw_paths}")
    print(f"processed_paths {gcgd.processed_paths}")
    print(f"len() {gcgd.len()}")
    gcgd.print_summary()
    print(f"num_node_features {gcgd.num_node_features}")
    print(f"num_edge_features {gcgd.num_edge_features}")
    # print(f"num_classes {gcgd.num_classes}")

    instance = gcgd.get(0)
    instance.print_summary()

    print("done test 1")




if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_test_1:
        test_1(args)
    if args.run_test_2:
        test_2(args)
