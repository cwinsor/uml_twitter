
'''

'''

import datetime
import math
import argparse
import logging
import wandb
from tqdm import trange

import json
import ijson
import torch

from GraphSSL.data import load_dataset, split_dataset, build_loader
from GraphSSL.model import Encoder
from GraphSSL.loss import infonce

# runtime arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, required=True,
                    help="data folder root")


class OriginalTweet():
    def __init__(self, original_tweet_id, original_tweet_text):
        self.original_tweet_id = original_tweet_id
        self.original_tweet_text = original_tweet_text
        self.retweet_list = []

    def __lt__(self,other):
        return self.original_tweet_id < other.original_tweet_id

    def add_retweet(self, retweet_user_id, retweet_date):
        self.retweet_list.append({"by_id": retweet_user_id, "date": retweet_date})

    def merge_from(self, other):
        self.retweet_list = self.retweet_list + other.retweet_list

    def toJson(self):
        the_data = {"tweet_id": {
            "tweet_text": self.original_tweet_text,
            "retweet_list": self.retweet_list
            }
            }
        json_object = json.dumps(the_data, indent=4)
        return json_object

def process_tweet_stream(tweet_stream, group_1_in, group_2_in, group_0_out, group_1_out):

    cutoff_1 = 1234
    cutoff_2 = math.inf

    current_group_1 = group_1_in.read()
    current_group_2 = group_2_in.read()
    current_input = tweet_stream.read()

    

            self.cutoff = cutoff


def create_initial_group_files(root_folder):

    def write_json(root_folder, file_name, breakpoint):
        fout_full_path = root_folder + "\\3_grouped\\" + "group_" + file_name
        f = open(fout_full_path, "w", encoding="utf-8")
        the_data = {"breakpoint": breakpoint}
        json_object = json.dumps(the_data, indent=4)
        f.write(json_object)

    write_json(root_folder, "group_0.jsonl", 1024)
    write_json(root_folder, "group_1.jsonl", 2048)
    write_json(root_folder, "group_2.jsonl", math.inf)


def parse_raw_data_file(root_folder):

    # read raw .ijson file
    # parse out the retweets
    # sort by original tweet ID
    # write to "parsed_and_sorted" file
    file_list = ["ids_geo_2020-02-01.jsonl",
                 "ids_geo_2020-02-02.jsonl"]
    for file_name in file_list:
        f_in_full_path = root_folder + "\\1_raw\\" + file_name
        with open(f_in_full_path, "r", encoding="utf-8") as f_in:
            print(f"processing {f_in_full_path}")
            tweets = ijson.items(f_in, "", multiple_values=True)

            all_data = {}
            for tweet in tweets:
                # if its a retweet
                if "retweeted_status" in tweet.keys():
                    original_tweet_id = tweet["retweeted_status"]["id"]
                    original_tweet_text = tweet["retweeted_status"]["full_text"]
                    retweet_user_id = tweet["user"]["id"]
                    retweet_date = tweet["created_at"]

                    if len(all_data) == 0 or original_tweet_id not in all_data.keys():
                        # initial entry for this original_tweet
                        all_data[original_tweet_id] = {"original_text": original_tweet_text,
                                                       "retweets": []}
                    all_data[original_tweet_id]["retweets"].append({"user": retweet_user_id,
                                                                    "date": retweet_date})
                    all_data[original_tweet_id]["num_retweets"] = len(all_data[original_tweet_id]["retweets"])
            all_data_sorted = dict(sorted(all_data.items()))

            fout_full_path = root_folder + "\\2_parsed\\" + "parsed_" + file_name
            f_out = open(fout_full_path, "w", encoding="utf-8")
            json_object = json.dumps(all_data_sorted, indent=4)
            f_out.write(json_object)

        print("done")

def add_new_data_to_group_files(root_folder):

    # open the N group files
    # get the "greakpoint" for each group file
    # initial "target" object for each is "None"

    # open new data file (previously sorted by original_tweet ID)
    # for each "current" object:

    # choose which group file and insert into that group file:

    #   target object is None:
    # 

    # working is None:
    # if A > B
    #   working = B
    # else
    #   working = A

    # if A < working
    # 

    # working is not None and (A > working and B > working)
    # write working to file and set Working to None

    #   create a new target object, initialize it, and hold target

    #   src object = target object
    #   add src object to target and hold target

    #   src object < target object
    #   write out current target object, get next target object and hold it

    

    # case 2: current_object_tweet_id > target_object tweet_id
    # repeat read/write target_object_tweets until target_object_tweet_id <


    # if target_data_file.original_tweet_id is None or read/echo data until:
    #    end of file
    #    ptr == new_file tweet_id
    #    ptr_tweet_id > new_file_tweet_id
    #      add new_file_tweet as new entry in ptr_tweet_id
    # 
    # for each entry in the new data file

    # parse out the retweets
    # sort by original tweet ID
    # write to "parsed_and_sorted" file

    new_data = root_folder + "\\raw\\ids_geo_2020-02-02.jsonl"
    with open(raw_data, "r", encoding="utf-8") as f:
        print("before")
        tweets = ijson.items(f, "", multiple_values=True)
        print("after")
        for tweet in tweets:
            pass
        print("done")





    # parser = ijson.parse(urlopen('http://.../'))
    # stream.write('<geo>')
    # for prefix, event, value in parser:
    #     if (prefix, event) == ('earth', 'map_key'):
    #         stream.write('<%s>' % value)
    #         continent = value
    #     elif prefix.endswith('.name'):
    #         stream.write('<object name="%s"/>' % value)
    #     elif (prefix, event) == ('earth.%s' % continent, 'end_map'):
    #         stream.write('</%s>' % continent)
    # stream.write('</geo>')

    # p
    # json.dumps(ind)


    # with open(folder_path + "file_1", "w", encoding="utf-8") as f:
    #     ijson.write(f, "", multiple_values=True)




# def process_large_files(args):

#     group_0 = open(args.data_src + group_0.ijson, "r")
#     group_1 = open(args.data_src + group_0.ijson, "r")

#     w = open("filepath", "w")
#     l = r.readline()
#     while l:
#         x = l.split(' ')[0]
#         y = l.split(' ')[1]
#         z = l.split(' ')[2]
#         w.write(l.replace(x,x[:-3]).replace(y,y[:-3]).replace(z,z[:-3]))
#         l = r.readline()
#     r.close()

def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    logger.info("start")

    original_tweet = OriginalTweet(original_tweet_id="123", original_tweet_text="abc")
    original_tweet.add_retweet(retweet_user_id="user1", retweet_date="date1")
    original_tweet.add_retweet(retweet_user_id="user1", retweet_date="date1")
    original_tweet.add_retweet(retweet_user_id="user1", retweet_date="date2")

    original_tweet2 = OriginalTweet(original_tweet_id="dont care", original_tweet_text="dont care")
    original_tweet2.add_retweet(retweet_user_id="user1", retweet_date="date1")
    original_tweet2.add_retweet(retweet_user_id="user1", retweet_date="date1")
    original_tweet2.add_retweet(retweet_user_id="user1", retweet_date="date2")

    original_tweet.merge_from(original_tweet2)

    file_name = "temp.jsonl"
    fout_full_path = args.data_root + "\\2_parsed\\" + "parsed_" + file_name
    f_out = open(fout_full_path, "w", encoding="utf-8")
    
    json_data = original_tweet.toJson()
    f_out.write(json_data)
    

    # create_initial_group_files(args.data_root)
    # parse_raw_data_file(args.data_root)
    # open_group_file_and_add_an_item_and_write(args.data_root)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
