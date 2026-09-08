import json
import os
from pathlib import Path
import tweepy
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
print("Looking for .env at:", env_path)
print("Exists:", env_path.exists())

if not env_path.exists():
    raise FileNotFoundError(
        f".env file not found:\n{env_path}"
    )

load_dotenv(env_path, override=True)
bearer_token = os.getenv("BEARER_TOKEN")
print("Bearer loaded:", bearer_token is not None)

if not bearer_token:
    raise ValueError("BEARER_TOKEN not found in .env")

# Create X API v2 client
client = tweepy.Client(
    bearer_token=bearer_token,
    wait_on_rate_limit=True
)

# Search recent tweets
try:
    response = client.search_recent_tweets(
        query="#Python",
        max_results=10,
        tweet_fields=[
            "created_at",
            "author_id",
            "lang",
            "public_metrics"
        ]
    )

    if response.data:

        with open("python.json", "w", encoding="utf-8") as f:

            for tweet in response.data:

                print("-" * 60)
                print(tweet.text)

                json.dump(tweet.data, f)
                f.write("\n")

        print(f"\nSaved {len(response.data)} tweets to python.json")

    else:
        print("No tweets found.")

except tweepy.TweepyException as e:
    print("Twitter API Error:")
    print(e)