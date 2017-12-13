from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import sys
from datetime import datetime
from datetime import timedelta
import got
from langdetect import detect
import praw

# Globals
NEWS_DOMAIN = "https://www.cryptocoinsnews.com/news/page/"

FILE_PATH = "news_and_tweets/"

RAW_HEADLINES = "raw/crypto_news_headlines.csv"
RAW_BTC_TWEETS = "raw/bitcoin_tweets.csv"
RAW_ETH_TWEETS = "raw/ethereum_tweets.csv"
RAW_LTC_TWEETS = "raw/litecoin_tweets.csv"

LABELED_HEADLINES = "labeled/headlines_labeled"
LABELED_BTC_TWEETS = "labeled/bitcoin_tweets_labeled"
LABELED_ETH_TWEETS = "labeled/ethereum_tweets_labeled"
LABELED_LTC_TWEETS = "labeled/litecoin_tweets_labeled"

BTC_PRICE_FILE = "coin_prices/bitcoin_price.csv"
ETH_PRICE_FILE = "coin_prices/ethereum_price.csv"
LTC_PRICE_FILE = "coin_prices/litecoin_price.csv"

TRAIN = "_train.csv"
DEV = "_dev.csv"
TEST = "_test.csv"

PERCENT_TRAIN = .60
PERCENT_DEV = .20
PERCENT_TEST = .20


############################################################################
# Get news headlines
############################################################################
def get_news(start_date, stop_date):

    # Initializations
    data = []
    page_num = 0
    stop = False

    # Loop through pages on website until date older than desired start date is found
    while not stop:

        # Scrape page
        page_num += 1
        r = requests.get(NEWS_DOMAIN + str(page_num) + "/", headers={'User-Agent': 'Mozilla/5.0'})
        c = r.content
        soup = BeautifulSoup(c, "html.parser")
        results = soup.find_all('div', {'class': 'grid-text'})
        
        # Parse data and store in list of dictionaries
        # ex: [ {'title': 'bitcoin up 1000', 'date': 27/09/2017}, ... ]
        for item in results:

            # Parse / clean data
            title = item.find("h3", {"class": "entry-title"}).text
            title = title.replace(",", "")
            title = title.replace("(+)", "")
            title = title.encode('utf-8')
            date = item.find("span", {"class": "date"}).text

            # Add to temp dictionary
            temp_dict = {}
            temp_dict['title'] = title
            temp_dict['date'] = datetime.strptime(date, '%d/%m/%Y').strftime('%m/%d/%Y')

            # If article is from after start_date and before stop_date, append to master list
            article_date = datetime.strptime(temp_dict['date'], '%m/%d/%Y')
            if article_date > stop_date:
                continue
            elif article_date >= start_date:
                data.append(temp_dict)
            else:
                stop = True
                break

        print "[%d] Last article date: %s" % (page_num, article_date)

    # Store the list of data in a Pandas dataframe
    results_df = pd.DataFrame(data)
    print("Scraped {} headlines".format(results_df.shape[0]))
    print(results_df.head())

    # Save to csv 
    results_df.to_csv(FILE_PATH + RAW_HEADLINES)

############################################################################
# Get tweets
############################################################################
def get_tweets(start_date, stop_date, num_daily):

	# Setup
    numDailyTweets = num_daily
    coins = ['bitcoin', 'ethereum', 'litecoin']

	# Loop through each coin
    for coin in coins:

        # Setup
        data = []

        # Loop through desired dates
        current_date = stop_date
        while current_date >= start_date:

            # Change datetime object to string
            date1 = current_date.strftime("%Y-%m-%d")
            date2 = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

            # For each date, need to ensure sufficient number of tweets are obtained (since non-English ones are thrown out)
            target_hit = False
            num_tries = -1
            count = 0
            while not target_hit and num_tries <= 3:
                
                # Set number of tweets to get
                num_tries += 1
                extra = num_tries * 50
                numDailyTweets = num_daily + extra

                # Get tweets
                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(coin).setSince(date1).setUntil(date2).setMaxTweets(numDailyTweets) 
                got_tweets = False
                while not got_tweets:
                    try:  
                        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
                        got_tweets = True
                    except:
                        continue

                # Determine where in tweets to start
                if num_tries == 0: start = 0
                else: start = num_daily + extra - 50

                # Add each one to master list
                for tweet in tweets[start:]:
                    if count >= num_daily: break
                    # Make sure tweet is in English
                    try: is_english = detect(tweet.text) == 'en'
                    except: is_english = False
                    if is_english:
                        temp_dict = {}
                        # Do some cleaning of the tweet text
                        tweet_text_split = tweet.text.replace(",", "").encode('utf-8').split()
                        to_remove = ['http://', 'http://www.', 'https://', 'https://www.', '...']
                        for word in tweet_text_split:
                            if '/' in word: to_remove.append(word)
                        for item in to_remove:
                            if item in tweet_text_split: tweet_text_split.remove(item)
                        temp_dict['tweet'] = ' '.join(tweet_text_split)
                        # Add date and username
                        temp_dict['date'] = tweet.date.strftime('%m/%d/%Y')
                        temp_dict['username'] = tweet.username
                        # Append to temp_dict
                        data.append(temp_dict)
                        count += 1

                # Check if sufficient number of tweets have been added
                if count >= num_daily:
                    target_hit = True

            print "Just added %d %s tweets from %s [took %d tries]" % (count, coin, current_date, num_tries)

            # Update current_date
            current_date = current_date - timedelta(days=1)

        # Store the list of data in a Pandas dataframe
        results_df = pd.DataFrame(data)

        # Save to csv 
        if coin == 'bitcoin':
            results_df.to_csv(FILE_PATH + RAW_BTC_TWEETS)
        elif coin == 'ethereum':
            results_df.to_csv(FILE_PATH + RAW_ETH_TWEETS)
        else:
            results_df.to_csv(FILE_PATH + RAW_LTC_TWEETS)

############################################################################
# Calculate coin price changes (1 day and 2 days out)
############################################################################
def calc_price_changes(coin_df, start_date, stop_date):

    # Will return dict: keys are dates, values represent price change 1 day later and 2 days later
    # (1 means price goes up, 0 means price goes down)
    to_return = {}

    # Loop through price data
    for index, row in coin_df.iterrows():

        # Ignore 1st and 2nd date
        if index == 1 or index == 2:
            continue
        else:

            # Get date into correct format / break if start_date reached
            current_date = datetime.strptime(row['Date'], '%b %d, %Y')
            if current_date > stop_date:
                continue
            elif current_date >= start_date:
                date_str = current_date.strftime('%m/%d/%Y')
            else:
                break

            # Compute price change
            price_now = row['Close']
            price_one = coin_df.loc[index - 1]['Close']
            price_two = coin_df.loc[index - 2]['Close']
            change_one = 1 if (price_one - price_now) > 0 else 0
            change_two = 1 if (price_two - price_now) > 0 else 0

            # Add to dict
            to_return[date_str] = [change_one, change_two]

    return to_return

############################################################################
# Label news headlines / tweets
############################################################################
def label(subject, bitcoin_prices, ethereum_prices, litecoin_prices):

    ######################################################
    # Case 1: labeling news headlines
    ######################################################
    if subject == 'headlines':

        # Load saved news headlines
        headlines_df = pd.read_csv(FILE_PATH + RAW_HEADLINES)

        # Setup
        data = []

        # Loop through all news headlines
        for index, row in headlines_df.iterrows():

            # Will store data in temp dict, then append to master list
            temp_dict = {}

            # Grab headline info
            date = row['date']
            temp_dict['date'] = date
            temp_dict['headline'] = row['title']

            # Get price data for headline date
            try:
                temp_dict['bitcoin_one'] = bitcoin_prices[date][0]
                temp_dict['bitcoin_two'] = bitcoin_prices[date][1]
                temp_dict['ethereum_one'] = ethereum_prices[date][0]
                temp_dict['ethereum_two'] = ethereum_prices[date][1]
                temp_dict['litecoin_one'] = litecoin_prices[date][0]
                temp_dict['litecoin_two'] = litecoin_prices[date][1]
            except:
                continue 

            # Append to master list
            data.append(temp_dict)

        # Store the list of data in a Pandas dataframe, reverse order
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[::-1].reset_index(drop=True)
        results_df = results_df.reindex(columns=["date", "headline", "bitcoin_one", "bitcoin_two", \
                                        "ethereum_one", "ethereum_two", "litecoin_one", "litecoin_two"])

        # Determine train, dev, and test set indices    
        num_headlines = results_df.shape[0]

        # Train
        end_index = int(num_headlines * PERCENT_TRAIN)
        end_date = results_df.loc[end_index, 'date']
        next_date = end_date
        while next_date == end_date:
            end_index += 1
            next_date = results_df.loc[end_index, 'date']
        start_train_index = 0
        end_train_index = end_index
        # Dev
        end_index = int(num_headlines * (PERCENT_TRAIN + PERCENT_DEV))
        end_date = results_df.loc[end_index, 'date']
        next_date = end_date
        while next_date == end_date:
            end_index += 1
            next_date = results_df.loc[end_index, 'date']
        start_dev_index = end_train_index + 1
        end_dev_index = end_index
        # Test
        start_test_index = end_dev_index + 1
        end_test_index = num_headlines

        # Split dataframe
        train = results_df[results_df.index.isin(range(start_train_index,end_train_index))]
        dev = results_df[results_df.index.isin(range(start_dev_index,end_dev_index))]
        test = results_df[results_df.index.isin(range(start_test_index,end_test_index))]

        # Save to csv files
        train.to_csv(FILE_PATH + LABELED_HEADLINES + TRAIN)
        dev.to_csv(FILE_PATH + LABELED_HEADLINES + DEV)
        test.to_csv(FILE_PATH + LABELED_HEADLINES + TEST)

    ######################################################
    # Case 2: labeling tweets
    ######################################################
    else:

        # Loop through each coin
        #coins = ['bitcoin', 'ethereum', 'litecoin']
        coins = ['litecoin']

        for coin in coins:

            # Load saved tweets
            if coin == 'bitcoin':
                tweets_df = pd.read_csv(FILE_PATH + RAW_BTC_TWEETS)
            elif coin == 'ethereum':
                tweets_df = pd.read_csv(FILE_PATH + RAW_ETH_TWEETS)
            else:
                tweets_df = pd.read_csv(FILE_PATH + RAW_LTC_TWEETS)

            # Setup
            data = []

            # Loop through all bitcoin tweets
            for index, row in tweets_df.iterrows():

                temp_dict = {}
                date = row['date']
                temp_dict['date'] = date
                temp_dict['tweet'] = row['tweet']

                if coin == 'bitcoin':
                    try:
                        temp_dict['bitcoin_one'] = bitcoin_prices[date][0]
                        temp_dict['bitcoin_two'] = bitcoin_prices[date][1]
                        data.append(temp_dict)
                    except:
                        continue
                elif coin == 'ethereum':
                    try:
                        temp_dict['ethereum_one'] = ethereum_prices[date][0]
                        temp_dict['ethereum_two'] = ethereum_prices[date][1]
                        data.append(temp_dict)
                    except:
                        continue
                else:
                    try:
                        temp_dict['litecoin_one'] = litecoin_prices[date][0]
                        temp_dict['litecoin_two'] = litecoin_prices[date][1]
                        data.append(temp_dict)
                    except:
                        continue

            # Store the list of data in a Pandas dataframe, reverse order
            results_df = pd.DataFrame(data)
            results_df = results_df.iloc[::-1].reset_index(drop=True)
            if coin == 'bitcoin':
                results_df = results_df.reindex(columns=["date", "tweet", "bitcoin_one", "bitcoin_two"])
            elif coin == 'ethereum':
                results_df = results_df.reindex(columns=["date", "tweet", "ethereum_one", "ethereum_two"])   
            else:
                results_df = results_df.reindex(columns=["date", "tweet", "litecoin_one", "litecoin_two"]) 

            # Determine train, dev, and test set indices    
            num_tweets = results_df.shape[0]

            # Train
            end_index = int(num_tweets * PERCENT_TRAIN)
            end_date = results_df.loc[end_index, 'date']
            next_date = end_date
            while next_date == end_date:
                end_index += 1
                next_date = results_df.loc[end_index, 'date']
            start_train_index = 0
            end_train_index = end_index
            # Dev
            end_index = int(num_tweets * (PERCENT_TRAIN + PERCENT_DEV))
            end_date = results_df.loc[end_index, 'date']
            next_date = end_date
            while next_date == end_date:
                end_index += 1
                next_date = results_df.loc[end_index, 'date']
            start_dev_index = end_train_index + 1
            end_dev_index = end_index
            # Test
            start_test_index = end_dev_index + 1
            end_test_index = num_tweets

            # Split dataframe
            train = results_df[results_df.index.isin(range(start_train_index,end_train_index))]
            dev = results_df[results_df.index.isin(range(start_dev_index,end_dev_index))]
            test = results_df[results_df.index.isin(range(start_test_index,end_test_index))]

            # Save to csv files
            if coin == 'bitcoin':
                train.to_csv(FILE_PATH + LABELED_BTC_TWEETS + TRAIN)
                dev.to_csv(FILE_PATH + LABELED_BTC_TWEETS + DEV)
                test.to_csv(FILE_PATH + LABELED_BTC_TWEETS + TEST)
            elif coin == 'ethereum':
                train.to_csv(FILE_PATH + LABELED_ETH_TWEETS + TRAIN)
                dev.to_csv(FILE_PATH + LABELED_ETH_TWEETS + DEV)
                test.to_csv(FILE_PATH + LABELED_ETH_TWEETS + TEST)
            else:
                train.to_csv(FILE_PATH + LABELED_LTC_TWEETS + TRAIN)
                dev.to_csv(FILE_PATH + LABELED_LTC_TWEETS + DEV)
                test.to_csv(FILE_PATH + LABELED_LTC_TWEETS + TEST)


############################################################################
# Main
############################################################################
def main():

    # Define start and stop dates
    start_date = datetime.strptime('01/01/2017', '%d/%m/%Y')
    stop_date = datetime.strptime('30/11/2017', '%d/%m/%Y')

    # Get news headlines
    if (True):
        get_news(start_date, stop_date)

    # Get tweets
    if (True):
        num_daily_tweets = 100
        get_tweets(start_date, stop_date, num_daily_tweets)

    # Load coin data, compute price changes
    if (True):
        bitcoin_df = pd.read_csv(BTC_PRICE_FILE)
        ethereum_df = pd.read_csv(ETH_PRICE_FILE)
        litecoin_df = pd.read_csv(LTC_PRICE_FILE)
        bitcoin_prices = calc_price_changes(bitcoin_df, start_date, stop_date)
        ethereum_prices = calc_price_changes(ethereum_df, start_date, stop_date)
        litecoin_prices = calc_price_changes(litecoin_df, start_date, stop_date)

    # Label data
    if (True):
        label('headlines', bitcoin_prices, ethereum_prices, litecoin_prices)
        label('tweets', bitcoin_prices, ethereum_prices, litecoin_prices)


if __name__ == '__main__':
	main()