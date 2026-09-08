import json
import os
from pathlib import Path
import tweepy
from dotenv import load_dotenv


# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

bearer_token = os.getenv("BEARER_TOKEN")

if not bearer_token:
    raise ValueError("BEARER_TOKEN missing")


# X API v2 client
client = tweepy.Client(
    bearer_token=bearer_token,
    wait_on_rate_limit=True
)


query = "#Python"

limit = 100

tweets = []


response = client.search_recent_tweets(
    query=query,
    max_results=100,
    tweet_fields=[
        "created_at",
        "author_id",
        "public_metrics"
    ],
    expansions=[
        "author_id"
    ]
)


if response.data:

    for tweet in response.data:

        data = {
            "id": tweet.id,
            "date": str(tweet.created_at),
            "content": tweet.text,
            "likes": tweet.public_metrics["like_count"],
            "retweets": tweet.public_metrics["retweet_count"]
        }

        tweets.append(data)

        print("-" * 60)
        print(tweet.created_at)
        print(tweet.text)


with open(
    "python.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        tweets,
        f,
        indent=4,
        ensure_ascii=False
    )


print(f"Saved {len(tweets)} tweets")