import json
import os
import tweepy
from tweepy import OAuthHandler, Stream
from dotenv import load_dotenv

from X_Data_Mining.data_mining import MyStreamListener

load_dotenv(
    dotenv_path=r"D:\ML Projects\machine_learning_projects\.env"
)
consumer_key = os.getenv('CONSUMER-KEY')
consumer_secret = os.getenv('CONSUMER-SECRET')
access_token = os.getenv('ACCESS-TOKEN')
access_secret = os.getenv('ACCESS-SECRET')

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#To read our own feed, number of tweets - 10
for status in tweepy.Cursor(api.home_timeline).items(10):
    #process_or_store(status._json) # our own feed
    #process_or_store(friend._json) # list of all our followers
    #process_or_store(tweet._json) # public tweets
    print(status.text)

def process_or_store(tweet):
    print(json.dumps(tweet))

twitter_stream = Stream(auth, MyStreamListener())
twitter_stream.filter(track=['#Python'])