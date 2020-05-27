import tweepy

# Authorization keys
api_Key = "HQCX3Z7gb1CFj5M6CS4NRzvd5"                     # your api key
api_Secret = "2YD5WYu8vamYQ3XD3pBXQoKCmsEbqCskBib3pTBqQtWk4IaVb1"                  # your api Secret
access_token = "352892839-YKIX8C40N5GpHKx9Z5RYaxd2lvwNK8oHQ0nRh2O2"
access_token_secret = "x6iBMiXsDReetAQvv2HhW91G1qRPvU0vGi3NWW2nsnqel"


def get_tweets(hashtag, language, resultType, n):
    # Authorization to api key and api secret
    auth = tweepy.OAuthHandler(api_Key, api_Secret)

    # Access to user's access token and access token secret
    auth.set_access_token(access_token, access_token_secret)

    # Calling api
    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search, q= '#namo', result_type=resultType, lang=language, tweet_mode='extended').items(n)

    # extended means all the tweets, else they will give truncated tweets.

    tweets_list = [tweet.full_text for tweet in tweets]
    return tweets_list

