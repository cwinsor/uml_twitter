
'''

'''
import os
import math
import argparse
import logging
from datetime import datetime

import numpy as np
import json
from json import JSONEncoder
import ijson

import matplotlib.pyplot as plt

# runtime arguments
parser = argparse.ArgumentParser()


parser.add_argument("--raw_folder", 
                    type=str, required=False,
                    default="D:\\dataset_covid_GeoCovGraph\\1_raw",
                    help="path to folder with raw .ijson files")

parser.add_argument("--parsed_folder", 
                    type=str, required=False,
                    default="D:\\dataset_covid_GeoCovGraph\\2_parsed",
                    help="path to folder with parsed .ijson files")

parser.add_argument("--merged_folder", 
                    type=str, required=False,
                    default="D:\\dataset_covid_GeoCovGraph\\3_merged",
                    help="path to the folder with the merged output")

# parse
parser.add_argument("--perform_parse",
                    default=False, action="store_true",
                    help="perform first step in preprocessing: parse the .ijson file")

parser.add_argument("--raw_file_in",
                    type=str, required=False,
                    default="_not_provided",
                    help="name of the .ijson raw file")

# merge
parser.add_argument("--perform_merge",
                    default=False, action="store_true",
                    help="perform second pass preprocessing which is: merge fresh tweets into prior tweet files")

parser.add_argument("--parsed_file_in",
                    type=str, required=False,
                    help="the fresh/new retweets")

parser.add_argument("--merged_file_in_out",
                    type=str, required=False,
                    help="mergeed file - read, merged with new tweets and re-written")

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


# subclass JSONEncoder
class TweetEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class IncomingTweet():
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
    tweet_obj = IncomingTweet()
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

    if args.perform_parse:
        # read raw .ijson file
        # find the retweets (eliminating replies).
        # capture the important information - original tweet ID, original tweet text, retweet date, (etc)
        # a standardizing class is defined above.
        # sort by original tweet ID
        # write to file "parsed_and_sorted"
        # this step is done in-memory, specifically the sort, and can be done so because the .ijson file is limited to
        # one day worth of tweets

        full_path_in = args.raw_folder + args.raw_file_in
        full_path_out = args.parsed_folder + "parsed_" + args.raw_file_in

        with open(full_path_in, "r", encoding="utf-8") as f_in:

            print(f"processing {full_path_in}")
            tweets = ijson.items(f_in, "", multiple_values=True)

            all_data = []
            for tweet in tweets:
                if IncomingTweet.is_retweet(tweet):
                    incoming_tweet = IncomingTweet()
                    incoming_tweet.from_raw_format(tweet)
                    all_data.append(incoming_tweet)
                # if len(all_data) > 3:
                #     break

            all_data_sorted = sorted(all_data)
            with open(full_path_out, "w", encoding="utf-8") as f_out:
                for o in all_data_sorted:
                    json_object = json.dumps(o, cls=TweetEncoder)
                    f_out.write(json_object)

    # second pass - merging
    # given a new day worth of tweets, preprocessed above
    # merge this into the existing 'group' files
    # a group file contains a range of tweets using tweet ID
    # 
    # both the new and group files use the standard format, and
    # both the new and group files are assumed sorted (in order of )
    # this step is performed via streaming

    if args.perform_merge:
        fresh_tweets_file = args.parsed_folder + args.parsed_file_in
        fresh_tweets_file_handle = open(fresh_tweets_file, "r", encoding="utf-8")
        fresh_tweets = ijson.items(fresh_tweets_file_handle, "", multiple_values=True)
        fresh_tweet = get_next(fresh_tweets)

        # move aside the current merge file adding suffix <datetime>, and open as read-source
        dst_filename = args.merged_folder + args.merged_file_in_out
        src_filename = f"{dst_filename}.{datetime.now().strftime('%m%d_%H%M%S')}"
        if not os.path.exists(dst_filename):
            print(f"creating new merge file")
            open(src_filename, 'x').close()
        else:
            print(f"moving prior merge file {dst_filename} to {src_filename}")
            os.rename(dst_filename, src_filename)

        merge_src_file_handle = open(src_filename, "r", encoding="utf-8")
        merge_src_tweets = ijson.items(merge_src_file_handle, "", multiple_values=True)
        merge_src_tweet = get_next(merge_src_tweets)

        f_out = open(dst_filename, "w", encoding="utf-8")

        while True:

            # print(f"group {prior_tweet.original_tweet_id} fresh {fresh_tweet.original_tweet_id}")

            if merge_src_tweet < fresh_tweet:
                # print("write from group, get next group")
                json_object = json.dumps(merge_src_tweet, cls=TweetEncoder)
                f_out.write(json_object)
                merge_src_tweet = get_next(merge_src_tweets)

            elif merge_src_tweet == fresh_tweet:
                # if both group and fresh are .inf flagged then we are done
                if merge_src_tweet.original_tweet_id == math.inf:
                    break
                # print("merge group into fresh, get next group")
                fresh_tweet.merge(merge_src_tweet)
                merge_src_tweet = get_next(merge_src_tweets)

            else:  # fresh_tweet < prior tweet
                # fresh tweet queue is ordered but not consolidated
                next_fresh_tweet = get_next(fresh_tweets)
                if next_fresh_tweet == fresh_tweet:
                    # merge
                    # print("merge fresh into fresh, get next fresh")
                    fresh_tweet.merge(next_fresh_tweet)
                else:
                    # print("write from fresh, get next fresh")
                    json_object = json.dumps(fresh_tweet, cls=TweetEncoder)
                    f_out.write(json_object)
                    fresh_tweet = get_next(fresh_tweets)

        fresh_tweets_file_handle.close()
        merge_src_file_handle.close()
        f_out.close()

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
