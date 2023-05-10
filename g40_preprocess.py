import argparse
import logging
import json
import ijson

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
        self.map_o_2_r = {}
        self.map_r_2_o = {}

    def parse_raw_tweet(self, raw):

        if "retweeted_status" not in raw.keys():
            return

        re_tweet_id = raw["id_str"]
        re_tweet_date = [raw["created_at"]]
        original_id = raw["retweeted_status"]["id_str"]
        original_text = raw["retweeted_status"]["full_text"]

        self.originals[original_id] = {"text": original_text}
        self.re_tweets[re_tweet_id] = {"date": re_tweet_date}

        if original_id not in self.map_o_2_r.keys():
            self.map_o_2_r[original_id] = []
        self.map_o_2_r[original_id].append(re_tweet_id)

        if re_tweet_id not in self.map_r_2_o.keys():
            self.map_r_2_o[re_tweet_id] = []
        self.map_r_2_o[re_tweet_id].append(original_id)


def main(args):

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"args: {args}")

    logger.info("start")

    if args.do_parse:

        parser = Parser()

        for filename in args.parse_file_list:
            print(f"processing {filename}")
            with open(args.parse_src_folder + filename, "r", encoding="utf-8") as f:
                raw_tweets = ijson.items(f, "", multiple_values=True)

                for raw_tweet in raw_tweets:
                    parser.parse_raw_tweet(raw_tweet)

        print("writing results...")

        with open(args.parse_dst_folder + "originals.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.originals, f, indent=4)
        with open(args.parse_dst_folder + "re_tweets.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.re_tweets, f, indent=4)
        with open(args.parse_dst_folder + "map_o_2_r.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.map_o_2_r, f, indent=4)
        with open(args.parse_dst_folder + "map_r_2_o.jsonl", "w", encoding="utf-8") as f:
            json.dump(parser.map_r_2_o, f, indent=4)

    # filter
    if args.do_filter:

        # make reduced/filtered list
        with open(args.filter_src_folder + "\\map_o_2_r.jsonl", "r", encoding="utf-8") as f:
            map_o_2_r = json.load(f)
        MIN_RETWEETS = 8
        MAX_RETWEETS = 8
        filtered_map_o_2_r = {k: v for k, v in map_o_2_r.items() if len(v) >= MIN_RETWEETS and len(v) <= MAX_RETWEETS}
        with open(args.filter_dst_folder + "\\filtered_map_o_2_r.jsonl", "w", encoding="utf-8") as f:
            json.dump(filtered_map_o_2_r, f, indent=4)

        # reconstruct the other tables
        with open(args.filter_src_folder + "\\originals.jsonl", "r", encoding="utf-8") as f:
            originals = json.load(f)
        with open(args.filter_src_folder + "\\re_tweets.jsonl", "r", encoding="utf-8") as f:
            re_tweets = json.load(f)
        with open(args.filter_src_folder + "\\map_r_2_o.jsonl", "r", encoding="utf-8") as f:
            map_r_2_o = json.load(f)

        filtered_map_r_2_o = {}
        filtered_originals = {}
        filtered_re_tweets = {}
        for original_id, retweet_list in filtered_map_o_2_r.items():
            filtered_originals[original_id] = originals[original_id]
            for retweet_id in retweet_list:
                filtered_re_tweets[retweet_id] = re_tweets[retweet_id]
                filtered_map_r_2_o[retweet_id] = original_id

        with open(args.filter_dst_folder + "\\filtered_originals.jsonl", "w", encoding="utf-8") as f:
            json.dump(filtered_originals, f, indent=4)
        with open(args.filter_dst_folder + "\\filtered_re_tweets.jsonl", "w", encoding="utf-8") as f:
            json.dump(filtered_re_tweets, f, indent=4)
        with open(args.filter_dst_folder + "\\filtered_map_r_2_o.jsonl", "w", encoding="utf-8") as f:
            json.dump(filtered_map_r_2_o, f, indent=4)

        print("done")

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
