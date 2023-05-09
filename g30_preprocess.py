import argparse
import logging
import json
import ijson

# runtime arguments
parser = argparse.ArgumentParser()

# parse
parser.add_argument("--perform_parse", default=False, action="store_true")
parser.add_argument("--daily_raw_folder", type=str, required=False)
parser.add_argument("--daily_parsed_folder", type=str, required=False)
parser.add_argument("--daily_file_list", nargs='+', type=str, required=False)

# merge
parser.add_argument("--perform_merge", default=False, action="store_true")
parser.add_argument("--merge_src_folder", type=str, required=False)
parser.add_argument("--merge_dst_folder", type=str, required=False)
# parser.add_argument("--merge_src_original_tweets", type=str, required=False, action='append')
parser.add_argument("--merge_file_list", nargs='+', type=str, required=False)


class RawTweet():
    def __init__(self, tweet):

        if "retweeted_status" not in tweet.keys():
            self.is_retweet = False
            return
        else:
            self.is_retweet = True

        self.original_tweet_id = tweet["retweeted_status"]["id_str"]
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
        def do_one_parse(filename):
            file_in = args.daily_raw_folder + filename
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

                file_out = args.daily_parsed_folder + "original_tweets_" + filename
                with open(file_out, "w", encoding="utf-8") as f_out:
                    temp = json.dumps(original_tweets, indent=4)
                    f_out.write(temp)
                f_out.close()

                file_out = args.daily_parsed_folder + "retweets_" + filename
                with open(file_out, "w", encoding="utf-8") as f_out:
                    temp = json.dumps(retweets, indent=4)
                    f_out.write(temp)
                f_out.close()

        for fname in args.daily_file_list:
            do_one_parse(fname)

    # merging
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

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
