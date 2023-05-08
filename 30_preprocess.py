import os
import math
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import json
from json import JSONEncoder
import ijson

import matplotlib.pyplot as plt

# runtime arguments
parser = argparse.ArgumentParser()

# parse
parser.add_argument("--perform_parse", default=False, action="store_true")
parser.add_argument("--daily_raw_folder", type=str, required=False)
parser.add_argument("--daily_parsed_folder", type=str, required=False)
parser.add_argument("--daily_raw_file_in", type=str, required=False)

# merge
parser.add_argument("--perform_merge", default=False, action="store_true")
parser.add_argument("--merge_src_folder", type=str, required=False)
parser.add_argument("--merge_dst_folder", type=str, required=False)
# parser.add_argument("--merge_src_original_tweets", type=str, required=False, action='append')
parser.add_argument("--merge_file_list", nargs='+', type=str, required=False)

# analyze
parser.add_argument("--perform_analyze",
                    default=False, action="store_true",
                    help="analyze_current_tweets")

parser.add_argument("--analyze_target",
                    type=str, required=False,
                    default="_not_provided",
                    help="analysis target - full path of file to be analyzed")
# filter
parser.add_argument("--perform_filter",
                    default=False, action="store_true",
                    help="filter_merged_tweets")

parser.add_argument("--filtered_folder_in", type=str, required=False)
parser.add_argument("--filtered_file_in", type=str, required=False)
parser.add_argument("--filtered_folder_out", type=str, required=False)
parser.add_argument("--filtered_file_out", type=str, required=False)


class RawTweet():
    def __init__(self, tweet):

        if "retweeted_status" not in tweet.keys():
            self.is_retweet = False
            return
        else:
            self.is_retweet = True

        self.original_tweet_id = tweet["retweeted_status"]["id"]
        self.original_tweet_data = {}
        self.original_tweet_data['text'] = tweet["retweeted_status"]["full_text"]

        self.retweet_id = tweet["id"]
        self.retweet_data = {}
        self.retweet_data['fk_original_tweet'] = self.original_tweet_id
        self.retweet_data['date'] = [tweet["created_at"]]


def main(args):

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"args: {args}")

    logger.info("start")

    if args.perform_parse:

        file_in = args.daily_raw_folder + args.daily_raw_file_in
        with open(file_in, "r", encoding="utf-8") as f_in:

            print(f"processing {file_in}")
            tweets = ijson.items(f_in, "", multiple_values=True)

            original_tweets = {}
            retweets = {}
            for tweet in tweets:
                in_tweet = RawTweet(tweet)
                if in_tweet.is_retweet:
                    original_tweets[in_tweet.original_tweet_id] = in_tweet.original_tweet_data
                    retweets[in_tweet.retweet_id] = in_tweet.retweet_data

            file_out = args.daily_parsed_folder + "original_tweets_" + args.daily_raw_file_in
            with open(file_out, "w", encoding="utf-8") as f_out:
                temp = json.dumps(original_tweets, indent=4)
                f_out.write(temp)
            f_out.close()

            file_out = args.daily_parsed_folder + "retweets_" + args.daily_raw_file_in
            with open(file_out, "w", encoding="utf-8") as f_out:
                temp = json.dumps(retweets, indent=4)
                f_out.write(temp)
            f_out.close()

    # second pass - merging
    # given a new day worth of tweets, preprocessed above
    # merge this into the existing 'group' files
    # a group file contains a range of tweets using tweet ID
    # 
    # both the new and group files use the standard format, and
    # both the new and group files are assumed sorted (in order of )
    # this step is performed via streaming

    if args.perform_merge:

        def do_original_tweets():
            original_tweets_all = {}
            for fname in args.merge_file_list:
                filepath = args.merge_src_folder + "\\original_tweets_" + fname
                print(f"processing {filepath}")
                with open(filepath, "r", encoding="utf-8") as myfile:
                    data = json.load(myfile)
                    # print(f"len data {len(data)}")
                    original_tweets_all.update(data)
                    # print(f"len {len(original_tweets_all)}")

            filepath = args.merge_dst_folder + "\\merged_original_tweets.json"
            with open(filepath, "w", encoding="utf-8") as f_out:
                temp = json.dumps(original_tweets_all, indent=4)
                f_out.write(temp)
            f_out.close()
        do_original_tweets()

        def do_retweets():
            retweets_all = {}
            for fname in args.merge_file_list:
                filepath = args.merge_src_folder + "\\retweets_" + fname
                print(f"processing {filepath}")
                with open(filepath, "r", encoding="utf-8") as myfile:
                    data = json.load(myfile)
                    # print(f"len data {len(data)}")
                    retweets_all.update(data)
                    # print(f"len {len(retweets_all)}")

            filepath = args.merge_dst_folder + "\\merged_retweets.json"
            with open(filepath, "w", encoding="utf-8") as f_out:
                temp = json.dumps(retweets_all, indent=4)
                f_out.write(temp)
            f_out.close()
        do_retweets()





    if args.perform_analyze:

        f = open(args.analyze_target, "r", encoding="utf-8")
        data = ijson.items(f, "", multiple_values=True)

        # for i in data:
        #     if i['number_retweets'] >= 150:
        #         # if i['number_retweets'] == 2476:
        #         for k, v in i.items():
        #             print(f"{k} {v}")
        #         assert False, "hold up"

        num_retweets_list = [original_tweet["number_retweets"] for original_tweet in data]

        num_retweets_list.sort(reverse=False)
        print(f"total original tweets = {len(num_retweets_list)}")
        # print(num_retweets_list[0:10])
        # print(num_retweets_list[-10:-1])

        nphist = np.histogram(num_retweets_list)
        print(nphist)
        # assert False, "hold up"

        plt.hist(x=num_retweets_list, bins=30, label="foo")
        plt.show()
        plt.hist(x=num_retweets_list, bins=30, range=[0.5, 30.5])
        plt.show()
        plt.hist(x=num_retweets_list, bins=30, range=[100, 7000])
        plt.show()

        f.close()

    if args.perform_filter:
        # filter the results for example only tweets having between a certain number of retweets
        src_filename = args.filtered_folder_in + args.filtered_file_in
        dst_filename = args.filtered_folder_out + args.filtered_file_out

        f_out = open(dst_filename, "w", encoding="utf-8")

        with open(src_filename, "r", encoding="utf-8") as f_in:

            print(f"processing {src_filename}")
            tweets = ijson.items(f_in, "", multiple_values=True)

            all_data = []
            for tweet in tweets:
                if tweet['number_retweets'] >= 8 and tweet['number_retweets'] <= 8:

                    json_object = json.dumps(tweet, cls=TweetEncoder)
                    f_out.write(json_object)


    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
