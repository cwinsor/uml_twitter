
'''
Dynamic topic modeling of GeoCoV19 using self-supervised graph learning

Step 2: training

This file trains the graph-based model using self-supervised learning.

Runtime parameters specify:
* prediction type (node, edge, graph)
* augmentation type (zona detail here)
* loss function (zona detail here)

Wandb is used for tracking.

The model is saved as <ZONA detail here> which can be used for downstream fine tuning.

Much code (that in /GraphSSL) follows and/or is copied from Maheshwari et al. with reference below.

References:
https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809
'''

import datetime
import math
import argparse
import logging
import wandb
from tqdm import trange

import json
from json import JSONEncoder
import ijson
import torch

from GraphSSL.data import load_dataset, split_dataset, build_loader
from GraphSSL.model import Encoder
from GraphSSL.loss import infonce

import heapq



# runtime arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, required=True,
                    help="data folder root")


# subclass JSONEncoder
class OriginalTweetEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class OriginalTweet():
    def __init__(self):
        self.original_tweet_id = math.inf
        self.original_tweet_text = "--NONE--"
        self.retweet_user_ids = []
        self.retweet_dates = []
        self.number_retweets = 0

    def is_retweet(tweet):
        return "retweeted_status" in tweet.keys()
    
    def from_raw_format(self, tweet):
        self.is_retweet = "retweeted_status" in tweet.keys()
        if self.is_retweet:
            self.original_tweet_id = tweet["retweeted_status"]["id"]
            self.original_tweet_text = tweet["retweeted_status"]["full_text"]
            self.retweet_user_ids = [tweet["user"]["id"]]
            self.retweet_dates = [tweet["created_at"]]
            self.number_retweets = 1

    def from_json_standard_format(self, the_data):
        # the_data = json.loads(the_data_in)
        # the_data = ijson.items(the_data_in, "", multiple_values=True)

        self.original_tweet_id = the_data["original_tweet_id"]
        self.original_tweet_text = the_data["original_tweet_text"]
        self.retweet_user_ids = the_data["retweet_user_ids"]
        self.retweet_dates = the_data["retweet_dates"]
        self.number_retweets = the_data["number_retweets"]

    def toJSON(self):
        the_data = {}
        the_data["original_tweet_id"] = self.original_tweet_id
        the_data["original_tweet_text"] = self.original_tweet_text
        the_data["retweet_user_ids"] = self.retweet_user_ids
        the_data["retweet_dates"] = self.retweet_dates
        the_data["number_retweets"] = self.number_retweets
        json_object = json.dumps(the_data, indent=4)
        return json_object

    def __lt__(self, other):
        return self.original_tweet_id < other.original_tweet_id
    def __gt__(self, other):
        return self.original_tweet_id > other.original_tweet_id
    def __eq__(self, other):
        return self.original_tweet_id == other.original_tweet_id
    

    def merge(self, other):
        self.retweet_user_ids += other.retweet_user_ids
        self.retweet_dates += other.retweet_dates
        self.number_retweets = len(self.retweet_user_ids)

# def read(filehandle):
#     #    the_data = ijson.items(the_data_in, "", multiple_values=True)

#     y = filehandle.read()
#     original_tweet = OriginalTweet()
#     original_tweet.from_json_standard_format(y)
#     return object


# def write(filehandle, object):
#     assert False, "zona not code complete"
#     for o in object:
#         json_object = json.dumps(object, cls=OriginalTweetEncoder)
#         filehandle.write(json_object)
#     # # json_object = ijson.items(the_data_in, "", multiple_values=True)
#     # json_object = ijson.items(the_data_in, "", multiple_values=True)
#     # ijson.
#     # filehandle.write(json_object)

def get_next(ijson_stream):
    tweet_obj = OriginalTweet()
    try:
        tweet_dict = next(ijson_stream)
        tweet_obj.from_json_standard_format(tweet_dict)
    except StopIteration:
        pass
    finally:
        return tweet_obj


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    logger.info("start")

    # # read raw .ijson file
    # # parse out the retweets
    # # sort by original tweet ID
    # # write to "parsed_and_sorted" file
    # group_file_list = ["ids_geo_2020-02-01.jsonl"]
    # for file_name in group_file_list:
    #     full_path_in = args.data_root + "\\1_raw\\" + file_name
    #     full_path_out = args.data_root + "\\2_selected_and_sorted\\" + "sns_" + file_name

    #     with open(full_path_in, "r", encoding="utf-8") as f_in:

    #         print(f"processing {full_path_in}")
    #         tweets = ijson.items(f_in, "", multiple_values=True)

    #         all_data = []
    #         for tweet in tweets:
    #             if OriginalTweet.is_retweet(tweet):
    #                 original_tweet = OriginalTweet()
    #                 original_tweet.from_raw_format(tweet)
    #                 all_data.append(original_tweet)
    #             if len(all_data) > 3:
    #                 break

    #         all_data_sorted = sorted(all_data)
    #         with open(full_path_out, "w", encoding="utf-8") as f_out:
    #             for o in all_data_sorted:
    #                 json_object = json.dumps(o, cls=OriginalTweetEncoder)
    #                 f_out.write(json_object)

    # # test...
    # group_file_list = ["sns_ids_geo_2020-02-01.jsonl"]
    # for file_name in group_file_list:
    #     full_path_new_tweets = args.data_root + "\\2_selected_and_sorted\\" + file_name
    #     with open(full_path_new_tweets, "r", encoding="utf-8") as f_in:
    #         print(f"processing {full_path_in}")

    #         tweets = ijson.items(f_in, "", multiple_values=True)

    #         tweet = next(tweets)
    #         print(tweet)
    #         tweet = next(tweets)
    #         print(tweet)
    #         tweet = next(tweets)
    #         print(tweet)
    #         tweet = next(tweets)
    #         print(tweet)

    # with open(full_path_group_out, "w", encoding="utf-8") as group_out_file: 
                    
    # second pass - merging
    full_path_fresh_tweets = args.data_root + "\\2_selected_and_sorted\\" + "sns_ids_geo_2020-02-01.jsonl"
    fresh_tweets_file_handle = open(full_path_fresh_tweets, "r", encoding="utf-8")
    fresh_tweets = ijson.items(fresh_tweets_file_handle, "", multiple_values=True)
    fresh_tweet = get_next(fresh_tweets)

    full_path_group_in = args.data_root + "\\3_grouped\\" + "group_1.jsonl"
    prior_tweets_file_handle = open(full_path_group_in, "r", encoding="utf-8")
    prior_tweets = ijson.items(prior_tweets_file_handle, "", multiple_values=True)
    prior_tweet = get_next(prior_tweets)

    full_path_group_out = args.data_root + "\\3_grouped\\" + "group_1b.jsonl"
    out_file = open(full_path_group_out, "w", encoding="utf-8")

    while True:

        print(f"group {prior_tweet.original_tweet_id} fresh {fresh_tweet.original_tweet_id}")

        if prior_tweet < fresh_tweet:
            print("write from group, get next group")
            json_object = json.dumps(prior_tweet, cls=OriginalTweetEncoder)
            out_file.write(json_object)
            prior_tweet = get_next(prior_tweets)

        elif prior_tweet == fresh_tweet:
            # if both group and fresh are .inf flagged then we are done
            if prior_tweet.original_tweet_id == math.inf:
                break
            print("merge group into fresh, get next group")
            fresh_tweet.merge(prior_tweet)
            prior_tweet = get_next(prior_tweets)

        else:
            print("write from fresh, get next fresh")
            json_object = json.dumps(fresh_tweet, cls=OriginalTweetEncoder)
            out_file.write(json_object)
            fresh_tweet = get_next(fresh_tweets)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
