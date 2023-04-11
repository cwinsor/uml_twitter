
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

import math
import argparse
import logging

import numpy as np
import json
from json import JSONEncoder
import ijson

import matplotlib.pyplot as plt

# runtime arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, required=True,
                    help="data folder root")

parser.add_argument("--perform_first_pass_parsing",
                    default=False, action="store_true",
                    help="perform first pass preprocessing which is: read fresh tweets from raw .ijson, select relevant data, write standard format")

parser.add_argument("--perform_second_pass_merging",
                    default=False, action="store_true",
                    help="perform second pass preprocessing which is: merge fresh tweets into prior tweet files")

parser.add_argument("--histogram_results",
                    default=False, action="store_true",
                    help="show histogram of retweet")


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

    if args.perform_first_pass_parsing:
        # read raw .ijson file
        # parse out the retweets
        # sort by original tweet ID
        # write to "parsed_and_sorted" file
        group_file_list = ["ids_geo_2020-02-01.jsonl"]
        for file_name in group_file_list:
            full_path_in = args.data_root + "\\1_raw\\" + file_name
            full_path_out = args.data_root + "\\2_selected_and_sorted\\" + "sns_" + file_name

            with open(full_path_in, "r", encoding="utf-8") as f_in:

                print(f"processing {full_path_in}")
                tweets = ijson.items(f_in, "", multiple_values=True)

                all_data = []
                for tweet in tweets:
                    if OriginalTweet.is_retweet(tweet):
                        original_tweet = OriginalTweet()
                        original_tweet.from_raw_format(tweet)
                        all_data.append(original_tweet)
                    # if len(all_data) > 3:
                    #     break

                all_data_sorted = sorted(all_data)
                with open(full_path_out, "w", encoding="utf-8") as f_out:
                    for o in all_data_sorted:
                        json_object = json.dumps(o, cls=OriginalTweetEncoder)
                        f_out.write(json_object)

    # second pass - merging
    if (args.perform_second_pass_merging):
        full_path_fresh_tweets = args.data_root + "\\2_selected_and_sorted\\" + "sns_ids_geo_2020-02-01.jsonl"
        fresh_tweets_file_handle = open(full_path_fresh_tweets, "r", encoding="utf-8")
        fresh_tweets = ijson.items(fresh_tweets_file_handle, "", multiple_values=True)
        fresh_tweet = get_next(fresh_tweets)

        full_path_group_in = args.data_root + "\\3_groups\\" + "group_1.jsonl"
        prior_tweets_file_handle = open(full_path_group_in, "r", encoding="utf-8")
        prior_tweets = ijson.items(prior_tweets_file_handle, "", multiple_values=True)
        prior_tweet = get_next(prior_tweets)

        full_path_group_out = args.data_root + "\\4_groups_updated\\" + "group_1.jsonl"  
        out_file = open(full_path_group_out, "w", encoding="utf-8")

        while True:

            # print(f"group {prior_tweet.original_tweet_id} fresh {fresh_tweet.original_tweet_id}")

            if prior_tweet < fresh_tweet:
                # print("write from group, get next group")
                json_object = json.dumps(prior_tweet, cls=OriginalTweetEncoder)
                out_file.write(json_object)
                prior_tweet = get_next(prior_tweets)

            elif prior_tweet == fresh_tweet:
                # if both group and fresh are .inf flagged then we are done
                if prior_tweet.original_tweet_id == math.inf:
                    break
                # print("merge group into fresh, get next group")
                fresh_tweet.merge(prior_tweet)
                prior_tweet = get_next(prior_tweets)

            else: # fresh_tweet < prior tweet
                # fresh tweet queue is ordered but not consolidated
                next_fresh_tweet = get_next(fresh_tweets)
                if next_fresh_tweet == fresh_tweet:
                    # merge
                    # print("merge fresh into fresh, get next fresh")
                    fresh_tweet.merge(next_fresh_tweet)
                else:
                    # print("write from fresh, get next fresh")
                    json_object = json.dumps(fresh_tweet, cls=OriginalTweetEncoder)
                    out_file.write(json_object)
                    fresh_tweet = get_next(fresh_tweets)

    if args.histogram_results:
        full_path_group_out = args.data_root + "\\4_groups_updated\\" + "group_1.jsonl"  
        f = open(full_path_group_out, "r")
        data = ijson.items(f, "", multiple_values=True)

        for i in data:
            if i['number_retweets'] == 10:
                for k, v in i.items():
                    print(f"{k} {v}")
                assert False, "hold up"

        num_retweets_list = [original_tweet["number_retweets"] for original_tweet in data]

        num_retweets_list.sort(reverse=False)
        print(len(num_retweets_list))
        # print(num_retweets_list[0:10])
        # print(num_retweets_list[-10:-1])

        nphist = np.histogram(num_retweets_list)
        for x in nphist:
            print(x)
        # print(nphist)

        plt.hist(num_retweets_list, 30, range=[0.5, 30.5])
        plt.show()
        plt.hist(num_retweets_list, 30, range=[100, 7000])
        plt.show()


        f.close()

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
