'''
G19HeteroDataset

References:
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset


'''

import argparse
import ijson

import torch
from torch_geometric.data import InMemoryDataset
from g19_hetero_data import G19HeteroData


# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_test_1", default=False, action="store_true")
parser.add_argument("--data_root", type=str, required=False, default="_not_provided")


class GeoCoV19GraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        data = self.get(idx)
        return data
        # y = [data.x_dict, data.edge_index_dict, data.edge_attr_dict]
        # return y

    # @property
    # def raw_file_names(self):
    #     return [self.raw_file_name]

    # def download(self):
    #     assert False, 'ERROR - download is not implemented'

    # @property
    # def processed_paths(self):
    #     return self.my_processed_paths

    @property
    def processed_file_names(self):
        return 'merged_all_out.jsonl'

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        f = open(self.root + "\\raw\\merged_all.jsonl", "r", encoding="utf-8")
        graphs_from_file = ijson.items(f, "", multiple_values=True)

        for graph_data_from_file in graphs_from_file:

            # if (graph_data_from_file["number_retweets"] > 20) and \
            #    (graph_data_from_file["number_retweets"] < 30):  # ZONA delete me
            if True:

                g19_hetero_data = G19HeteroData()

                g19_hetero_data.from_json_standard_format(ref=g19_hetero_data,
                                                          the_data=graph_data_from_file)

                # g19_hetero_data.print_summary()
                data_list.append(g19_hetero_data)

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
