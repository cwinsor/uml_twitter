from itertools import groupby
import argparse
import logging
import json
import ijson

from aaa_my_common import my_group_by

# runtime arguments
parser = argparse.ArgumentParser()

# parse
parser.add_argument("--do_parse", default=False, action="store_true")
parser.add_argument("--parse_src_folder", type=str, required=False)
parser.add_argument("--parse_dst_folder", type=str, required=False)
parser.add_argument("--parse_file_list", nargs='+', type=str, required=False)

# merge
parser.add_argument("--perform_merge", default=False, action="store_true")
parser.add_argument("--merge_src_folder", type=str, required=False)
parser.add_argument("--merge_dst_folder", type=str, required=False)
parser.add_argument("--merge_file_list", nargs='+', type=str, required=False)

# filter
parser.add_argument("--do_filter", default=False, action="store_true")
parser.add_argument("--filter_src_folder", type=str, required=False)
parser.add_argument("--filter_dst_folder", type=str, required=False)


class Parser():
    def __init__(self):

        self.originals = {}
        self.re_tweets = {}
        self.list_o_r = []
        # self.map_o_2_r = {}
        # self.map_r_2_o = {}

    def parse_raw_tweet(self, raw):

        if "retweeted_status" not in raw.keys():
            return

        re_tweet_id = raw["id_str"]
        re_tweet_date = [raw["created_at"]]
        original_id = raw["retweeted_status"]["id_str"]
        original_text = raw["retweeted_status"]["full_text"]

        self.originals[original_id] = {"text": original_text}
        self.re_tweets[re_tweet_id] = {"date": re_tweet_date}
        self.list_o_r.append((original_id, re_tweet_id))

    # groupby following https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
    def get_groupby_original(self):
        grouped = {}
        for k, group in groupby(self.list_o_r, lambda x: x[0]):
            grouped[k] = [item[1] for item in group]
            # print()
        return grouped

def main(args):

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"args: {args}")

    logger.info("start")

    if args.do_parse:

        parser = Parser()

        print("read raw files...")
        for filename in args.parse_file_list:
            print(f" {filename}")
            with open(args.parse_src_folder + filename, "r", encoding="utf-8") as f:
                raw_tweets = ijson.items(f, "", multiple_values=True)

                for raw_tweet in raw_tweets:
                    parser.parse_raw_tweet(raw_tweet)

        print("write results...")
        with open(args.parse_dst_folder + "originals.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.originals, f, indent=4)
        with open(args.parse_dst_folder + "re_tweets.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.re_tweets, f, indent=4)
        with open(args.parse_dst_folder + "list_o_r.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.list_o_r, f, indent=4)

        # print("generate map...")
        # o_2_r = parser.get_groupby_original()
        # print("write map...")
        # with open(args.parse_dst_folder + "map_o_2_r.jsonl", "w", encoding="utf-8") as f:
        #     json.dump(o_2_r, f, indent=4)

        print("done")

    # filter
    if args.do_filter:
        with open(args.filter_src_folder + "list_o_r.jsonl", "r", encoding="utf-8") as f:
            o2r = json.load(f)

        def first_element(x):
            return x[0]

        def second_element(x):
            return x[1]

        groups = my_group_by(o2r, first_element, second_element)

        MIN_RETWEETS = 6
        MAX_RETWEETS = 7
        keepset_originals = set()
        keepset_re_tweets = set()

        for key, grplist in groups.items():
            if len(grplist) >= MIN_RETWEETS and len(grplist) <= MAX_RETWEETS:
                keepset_originals.add(key)
                keepset_re_tweets.update(grplist)
       
        o2r_filtered = [item for item in o2r if item[0] in keepset_originals and item[1] in keepset_re_tweets]

        print("after filtering:")
        print(f"number of original tweets: {len(keepset_originals)}")
        print(f"number of retweets: {len(keepset_re_tweets)}")
        print(f"number of edges: {len(o2r_filtered)}")

        with open(args.filter_src_folder + "originals.jsonl", "r", encoding="utf-8") as f:
            originals = json.load(f)
        originals_filtered = {k: originals[k] for k in originals if k in keepset_originals}

        with open(args.filter_src_folder + "re_tweets.jsonl", "r", encoding="utf-8") as f:
            re_tweets = json.load(f)
        re_tweets_filtered = {k: re_tweets[k] for k in re_tweets if k in keepset_re_tweets}

        print("write results...")
        with open(args.filter_dst_folder + "originals_filtered.jsonl", "w", encoding="utf-8") as f:
            json.dump(originals_filtered, f, indent=4)
        with open(args.filter_dst_folder + "re_tweets_filtered.jsonl", "w", encoding="utf-8") as f:
            json.dump(re_tweets_filtered, f, indent=4)
        with open(args.filter_dst_folder + "list_o_r_filtered.jsonl", "w", encoding="utf-8") as f:
            json.dump(o2r_filtered, f, indent=4)

        print("done")


    # # filter
    # if args.do_filterxxxxx:

    #     # make reduced/filtered list
    #     with open(args.filter_src_folder + "\\map_o_2_r.jsonl", "r", encoding="utf-8") as f:
    #         map_o_2_r = json.load(f)
    #     MIN_RETWEETS = 8
    #     MAX_RETWEETS = 8
    #     filtered_map_o_2_r = {k: v for k, v in map_o_2_r.items() if len(v) >= MIN_RETWEETS and len(v) <= MAX_RETWEETS}
    #     with open(args.filter_dst_folder + "\\filtered_map_o_2_r.jsonl", "w", encoding="utf-8") as f:
    #         json.dump(filtered_map_o_2_r, f, indent=4)

    #     # reconstruct the other tables
    #     with open(args.filter_src_folder + "\\originals.jsonl", "r", encoding="utf-8") as f:
    #         originals = json.load(f)
    #     with open(args.filter_src_folder + "\\re_tweets.jsonl", "r", encoding="utf-8") as f:
    #         re_tweets = json.load(f)
    #     with open(args.filter_src_folder + "\\map_r_2_o.jsonl", "r", encoding="utf-8") as f:
    #         map_r_2_o = json.load(f)

    #     filtered_map_r_2_o = {}
    #     filtered_originals = {}
    #     filtered_re_tweets = {}
    #     for original_id, retweet_list in filtered_map_o_2_r.items():
    #         filtered_originals[original_id] = originals[original_id]
    #         for retweet_id in retweet_list:
    #             filtered_re_tweets[retweet_id] = re_tweets[retweet_id]
    #             filtered_map_r_2_o[retweet_id] = original_id

    #     with open(args.filter_dst_folder + "\\filtered_originals.jsonl", "w", encoding="utf-8") as f:
    #         json.dump(filtered_originals, f, indent=4)
    #     with open(args.filter_dst_folder + "\\filtered_re_tweets.jsonl", "w", encoding="utf-8") as f:
    #         json.dump(filtered_re_tweets, f, indent=4)
    #     with open(args.filter_dst_folder + "\\filtered_map_r_2_o.jsonl", "w", encoding="utf-8") as f:
    #         json.dump(filtered_map_r_2_o, f, indent=4)

    #     print("done")

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
