import pandas as pd
import numpy as np

from baseline_utility_229 import *

#################################################################
# Setup constant values to be used later 
#################################################################

# Labeled data file paths (train and dev)
HEADLINES_TRAIN_FILE = "headlines-and-tweets/labeled/headlines_labeled_train.csv"
HEADLINES_DEV_FILE = "headlines-and-tweets/labeled/headlines_labeled_dev.csv"
BTC_TRAIN_FILE = "headlines-and-tweets/labeled/bitcoin_tweets_labeled_train.csv"
BTC_DEV_FILE = "headlines-and-tweets/labeled/bitcoin_tweets_labeled_dev.csv"
ETH_TRAIN_FILE = "headlines-and-tweets/labeled/ethereum_tweets_labeled_train.csv"
ETH_DEV_FILE = "headlines-and-tweets/labeled/ethereum_tweets_labeled_dev.csv"
LTC_TRAIN_FILE = "headlines-and-tweets/labeled/litecoin_tweets_labeled_train.csv"
LTC_DEV_FILE = "headlines-and-tweets/labeled/litecoin_tweets_labeled_dev.csv"

# Column names (for pulling data out of csv files)
HEADLINE_TEXT_COLUMN = 'headline'
TWEET_TEXT_COLUMN = 'tweet'
BTC_COL_1_D, BTC_COL_2_D = 'bitcoin_one', 'bitcoin_two'
ETH_COL_1_D, ETH_COL_2_D = 'ethereum_one', 'ethereum_two'
LTC_COL_1_D, LTC_COL_2_D = 'litecoin_one', 'litecoin_two'

############################################################################
# Main
############################################################################
def main():

    #################################
    # Read in tweets
    #################################

    btc_tweets_train = pd.read_csv(BTC_TRAIN_FILE, index_col=0)
    btc_tweets_dev = pd.read_csv(BTC_DEV_FILE, index_col=0)
    eth_tweets_train = pd.read_csv(ETH_TRAIN_FILE, index_col=0)
    eth_tweets_dev = pd.read_csv(ETH_DEV_FILE, index_col=0)
    ltc_tweets_train = pd.read_csv(LTC_TRAIN_FILE, index_col=0)
    ltc_tweets_dev = pd.read_csv(LTC_DEV_FILE, index_col=0)

    #################################
    # Experiment 0: Get info about data
    #################################
    if (True):

        # Number
        btc_num_train = count_num_entries(btc_tweets_train)
        btc_num_dev = count_num_entries(btc_tweets_dev)
        eth_num_train = count_num_entries(eth_tweets_train)
        eth_num_dev = count_num_entries(eth_tweets_dev)
        ltc_num_train = count_num_entries(ltc_tweets_train)
        ltc_num_dev = count_num_entries(ltc_tweets_dev)
        print_num_entries_tweets(btc_num_train, btc_num_dev, eth_num_train, eth_num_dev, ltc_num_train, ltc_num_dev)

        # Label distribution
        btc_labels_train = count_labels(btc_tweets_train, 'btc_tweets')
        btc_labels_dev = count_labels(btc_tweets_dev, 'btc_tweets')
        eth_labels_train = count_labels(eth_tweets_train, 'eth_tweets')
        eth_labels_dev = count_labels(eth_tweets_dev, 'eth_tweets')
        ltc_labels_train = count_labels(ltc_tweets_train, 'ltc_tweets')
        ltc_labels_dev = count_labels(ltc_tweets_dev, 'ltc_tweets')
        print_labels_tweets(btc_labels_train, btc_labels_dev, eth_labels_train, eth_labels_dev, ltc_labels_train, ltc_labels_dev)

    #################################
    # Experiment 1: Try all different types of classifiers
    #################################
    if (False):

        classifiers = ['logistic_regression', 'linear_svc', 'multinomial_nb', 'bernoulli_nb']
        for classifier in classifiers:

            btc1_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_1_D, classifier)
            btc2_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_2_D, classifier)
            eth1_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_1_D, classifier)
            eth2_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_2_D, classifier)
            ltc1_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_1_D, classifier)
            ltc2_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_2_D, classifier)

            print "\n%s:" % classifier
            print "BTC1: %.2f%%, BTC2: %.2f%%, ETH1: %.2f%%, ETH2: %.2f%%, LTC1: %.2f%%, LTC2: %.2f%%" % \
                ((btc1_dev_results['accuracy'] * 100), (btc2_dev_results['accuracy'] * 100), (eth1_dev_results['accuracy'] * 100), 
                (eth2_dev_results['accuracy'] * 100), (ltc1_dev_results['accuracy'] * 100), (ltc2_dev_results['accuracy'] * 100))

    #################################
    # Experiment 2: Looking at feature weights
    #################################
    if (False):

        btc1_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_1_D, 'logistic_regression')
        btc2_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_2_D, 'logistic_regression')
        eth1_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_1_D, 'logistic_regression')
        eth2_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_2_D, 'logistic_regression')
        ltc1_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_1_D, 'logistic_regression')
        ltc2_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_2_D, 'logistic_regression')

        see_top_weights('Tweets - BTC 1 Day', btc1_dev_results['classifier'], btc1_dev_results['feature-names'], top_features=30)
        see_top_weights('Tweets - BTC 2 Days', btc2_dev_results['classifier'], btc2_dev_results['feature-names'], top_features=30)
        see_top_weights('Tweets - ETH 1 Day', eth1_dev_results['classifier'], eth1_dev_results['feature-names'], top_features=30)
        see_top_weights('Tweets - ETH 2 Days', eth2_dev_results['classifier'], eth2_dev_results['feature-names'], top_features=30)
        see_top_weights('Tweets - LTC 1 Day', ltc1_dev_results['classifier'], ltc1_dev_results['feature-names'], top_features=30)
        see_top_weights('Tweets - LTC 2 Days', ltc2_dev_results['classifier'], ltc2_dev_results['feature-names'], top_features=30)
		
    #################################
    # Experiment 3: Try out new evaluation metric for making single overall prediction each day
    #################################
    if (False):

        classifiers = ['logistic_regression', 'linear_svc', 'multinomial_nb', 'bernoulli_nb']
        for classifier in classifiers:

            btc1_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_1_D, classifier)
            btc2_dev_results = train_and_dev(btc_tweets_train, btc_tweets_dev, 'tweets', BTC_COL_2_D, classifier)
            eth1_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_1_D, classifier)
            eth2_dev_results = train_and_dev(eth_tweets_train, eth_tweets_dev, 'tweets', ETH_COL_2_D, classifier)
            ltc1_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_1_D, classifier)
            ltc2_dev_results = train_and_dev(ltc_tweets_train, ltc_tweets_dev, 'tweets', LTC_COL_2_D, classifier)

            print "\n%s:" % classifier
            print "BTC1: %.2f%%, BTC2: %.2f%%, ETH1: %.2f%%, ETH2: %.2f%%, LTC1: %.2f%%, LTC2: %.2f%%" % \
                ((btc1_dev_results['daily_accuracy'] * 100), (btc2_dev_results['daily_accuracy'] * 100), (eth1_dev_results['daily_accuracy'] * 100), 
                (eth2_dev_results['daily_accuracy'] * 100), (ltc1_dev_results['daily_accuracy'] * 100), (ltc2_dev_results['daily_accuracy'] * 100))


if __name__ == '__main__':
	main()
