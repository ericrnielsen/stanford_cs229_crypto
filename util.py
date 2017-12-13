import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import string

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score

################################################################################################
################################################################################################
# Setup constant values to be used later 
################################################################################################
################################################################################################

# Coins
COINS = ['bitcoin', 'ethereum', 'litecoin']

# Labeled data file paths (train and dev)
HEADLINES_TRAIN_FILE = "news_and_tweets_data/labeled/headlines_labeled_train.csv"
HEADLINES_DEV_FILE = "news_and_tweets_data/labeled/headlines_labeled_dev.csv"
BTC_TRAIN_FILE = "news_and_tweets_data/labeled/bitcoin_tweets_labeled_train.csv"
BTC_DEV_FILE = "news_and_tweets_data/labeled/bitcoin_tweets_labeled_dev.csv"
ETH_TRAIN_FILE = "news_and_tweets_data/labeled/ethereum_tweets_labeled_train.csv"
ETH_DEV_FILE = "news_and_tweets_data/labeled/ethereum_tweets_labeled_dev.csv"
LTC_TRAIN_FILE = "news_and_tweets_data/labeled/litecoin_tweets_labeled_train.csv"
LTC_DEV_FILE = "news_and_tweets_data/labeled/litecoin_tweets_labeled_dev.csv"

# Column names (for pulling data out of csv files)
DATE = 'date'
HEADLINE_TEXT_COLUMN = 'headline'
TWEET_TEXT_COLUMN = 'tweet'
BTC_COL_1_D, BTC_COL_2_D = 'bitcoin_one', 'bitcoin_two'
ETH_COL_1_D, ETH_COL_2_D = 'ethereum_one', 'ethereum_two'
LTC_COL_1_D, LTC_COL_2_D = 'litecoin_one', 'litecoin_two'
PREDICTION_LABELS = [BTC_COL_1_D, BTC_COL_2_D, ETH_COL_1_D, ETH_COL_2_D, LTC_COL_1_D, LTC_COL_2_D]

# Train, dev, and test dates
TRAIN_START_DATE = '2017-01-01'
TRAIN_END_DATE = '2017-07-19'
DEV_START_DATE = '2017-07-20'
DEV_END_DATE = '2017-09-23'
TEST_START_DATE = '2017-09-24'
TEST_END_DATE = '2017-11-30'
TIME_PERIODS = [(TRAIN_START_DATE, TRAIN_END_DATE), (DEV_START_DATE, DEV_END_DATE), (TEST_START_DATE, TEST_END_DATE)]

# Parser and punctuation
parser = spacy.load('en_core_web_sm')
punctuations = string.punctuation

################################################################################################
################################################################################################
# Loading datasets and calculating / printing basic info about them
################################################################################################
################################################################################################

#################################################################
# Read in news headlines or tweet csv file; return a pandas DataFrame
#################################################################
def load_csv(file_path, text_column):
    # Read from csv file, index by date, get rid of old index column if present, remove duplicates
    new_df = pd.read_csv(file_path, index_col=['date'], parse_dates=True)
    if 'Unnamed: 0' in new_df.columns: new_df = new_df.drop('Unnamed: 0', axis=1)
    new_df = new_df.drop_duplicates(subset=[text_column])
    return new_df

#################################################################
# Count number of entries in dataset
#################################################################
def count_num_entries(df):
    return df.shape[0]

#################################################################
# Print number of entries in datasets
#################################################################
def print_num_entries_headlines(hl_counts):
    print "\nTOTAL ENTRY COUNTS:"
    print "[Headlines]\t\t Total train entries: %d, Total dev entries: %d, Total test entries: %d" % (hl_counts[0], hl_counts[1], hl_counts[2])

def print_num_entries_tweets(btc_counts, eth_counts, ltc_counts):
    print "\nTOTAL ENTRY COUNTS:"
    print "[Bitcoin tweets]\t Total train entries: %d, Total dev entries: %d, Total test entries: %d" % (btc_counts[0], btc_counts[1], btc_counts[2])
    print "[Ethereum tweets]\t Total train entries: %d, Total dev entries: %d, Total test entries: %d" % (eth_counts[0], eth_counts[1], eth_counts[2])
    print "[Litcoin tweets]\t Total train entries: %d, Total dev entries: %d, Total test entries: %d" % (ltc_counts[0], ltc_counts[1], ltc_counts[2])

#################################################################
# Determine number of entries labeled '1' and '0' in dataset
#################################################################
def count_labels(df, data_type):
    toReturn = {}
    # Bitcoin tweet counts
    if data_type == 'btc_tweets':
        btc1_indices = df[BTC_COL_1_D].value_counts().index.tolist()
        btc1_counts = df[BTC_COL_1_D].value_counts().tolist()
        btc2_indices = df[BTC_COL_2_D].value_counts().index.tolist()
        btc2_counts = df[BTC_COL_2_D].value_counts().tolist()
        toReturn['btc1'] = (btc1_counts[0], btc1_counts[1]) if btc1_indices[0] == 0 else (btc1_counts[1], btc1_counts[0])
        toReturn['btc2'] = (btc2_counts[0], btc2_counts[1]) if btc2_indices[0] == 0 else (btc2_counts[1], btc2_counts[0])
     # Ethereum tweet counts
    elif data_type == 'eth_tweets':
        eth1_indices = df[ETH_COL_1_D].value_counts().index.tolist()
        eth1_counts = df[ETH_COL_1_D].value_counts().tolist()
        eth2_indices = df[ETH_COL_2_D].value_counts().index.tolist()
        eth2_counts = df[ETH_COL_2_D].value_counts().tolist()
        toReturn['eth1'] = (eth1_counts[0], eth1_counts[1]) if eth1_indices[0] == 0 else (eth1_counts[1], eth1_counts[0])
        toReturn['eth2'] = (eth2_counts[0], eth2_counts[1]) if eth2_indices[0] == 0 else (eth2_counts[1], eth2_counts[0])
    # Litecoin tweet counts
    elif data_type == 'ltc_tweets':
        ltc1_indices = df[LTC_COL_1_D].value_counts().index.tolist()
        ltc1_counts = df[LTC_COL_1_D].value_counts().tolist()
        ltc2_indices = df[LTC_COL_2_D].value_counts().index.tolist()
        ltc2_counts = df[LTC_COL_2_D].value_counts().tolist()
        toReturn['ltc1'] = (ltc1_counts[0], ltc1_counts[1]) if ltc1_indices[0] == 0 else (ltc1_counts[1], ltc1_counts[0])
        toReturn['ltc2'] = (ltc2_counts[0], ltc2_counts[1]) if ltc2_indices[0] == 0 else (ltc2_counts[1], ltc2_counts[0])
    # Headlines counts
    else:
        btc1_indices = df[BTC_COL_1_D].value_counts().index.tolist()
        btc1_counts = df[BTC_COL_1_D].value_counts().tolist()
        btc2_indices = df[BTC_COL_2_D].value_counts().index.tolist()
        btc2_counts = df[BTC_COL_2_D].value_counts().tolist()
        toReturn['btc1'] = (btc1_counts[0], btc1_counts[1]) if btc1_indices[0] == 0 else (btc1_counts[1], btc1_counts[0])
        toReturn['btc2'] = (btc2_counts[0], btc2_counts[1]) if btc2_indices[0] == 0 else (btc2_counts[1], btc2_counts[0])   
        eth1_indices = df[ETH_COL_1_D].value_counts().index.tolist()
        eth1_counts = df[ETH_COL_1_D].value_counts().tolist()
        eth2_indices = df[ETH_COL_2_D].value_counts().index.tolist()
        eth2_counts = df[ETH_COL_2_D].value_counts().tolist()
        toReturn['eth1'] = (eth1_counts[0], eth1_counts[1]) if eth1_indices[0] == 0 else (eth1_counts[1], eth1_counts[0])
        toReturn['eth2'] = (eth2_counts[0], eth2_counts[1]) if eth2_indices[0] == 0 else (eth2_counts[1], eth2_counts[0])
        ltc1_indices = df[LTC_COL_1_D].value_counts().index.tolist()
        ltc1_counts = df[LTC_COL_1_D].value_counts().tolist()
        ltc2_indices = df[LTC_COL_2_D].value_counts().index.tolist()
        ltc2_counts = df[LTC_COL_2_D].value_counts().tolist()
        toReturn['ltc1'] = (ltc1_counts[0], ltc1_counts[1]) if ltc1_indices[0] == 0 else (ltc1_counts[1], ltc1_counts[0])
        toReturn['ltc2'] = (ltc2_counts[0], ltc2_counts[1]) if ltc2_indices[0] == 0 else (ltc2_counts[1], ltc2_counts[0])
    # Return
    return toReturn

#################################################################
# Print number of entries labeled '1' and '0' in headline datasets
#################################################################
def print_labels_headlines(hl_labels):
    # Split input
    hl_labels_train, hl_labels_dev, hl_labels_test = hl_labels[0], hl_labels[1], hl_labels[2]
    # Get total number of entries in each dataset
    hl_num_train = hl_labels_train['btc1'][0] + hl_labels_train['btc1'][1]
    hl_num_dev = hl_labels_dev['btc1'][0] + hl_labels_dev['btc1'][1]
    hl_num_test = hl_labels_test['btc1'][0] + hl_labels_test['btc1'][1]
    # Print
    print "\nLABEL COUNTS:"
    print "[Headlines]\t\t Train +1 day BTC labeled 0: %d (%.2f%%)" % (hl_labels_train['btc1'][0], (hl_labels_train['btc1'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +1 day BTC labeled 1: %d (%.2f%%)" % (hl_labels_train['btc1'][1], (hl_labels_train['btc1'][1]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days BTC labeled 0: %d (%.2f%%)" % (hl_labels_train['btc2'][0], (hl_labels_train['btc2'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days BTC labeled 1: %d (%.2f%%)" % (hl_labels_train['btc2'][1], (hl_labels_train['btc2'][1]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +1 day ETH labeled 0: %d (%.2f%%)" % (hl_labels_train['eth1'][0], (hl_labels_train['eth1'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +1 day ETH labeled 1: %d (%.2f%%)" % (hl_labels_train['eth1'][1], (hl_labels_train['eth1'][1]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days ETH labeled 0: %d (%.2f%%)" % (hl_labels_train['eth2'][0], (hl_labels_train['eth2'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days ETH labeled 1: %d (%.2f%%)" % (hl_labels_train['eth2'][1], (hl_labels_train['eth2'][1]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +1 day LTC labeled 0: %d (%.2f%%)" % (hl_labels_train['ltc1'][0], (hl_labels_train['ltc1'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +1 day LTC labeled 1: %d (%.2f%%)" % (hl_labels_train['ltc1'][1], (hl_labels_train['ltc1'][1]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days LTC labeled 0: %d (%.2f%%)" % (hl_labels_train['ltc2'][0], (hl_labels_train['ltc2'][0]*100.)/hl_num_train)
    print "[Headlines]\t\t Train +2 days LTC labeled 1: %d (%.2f%%)" % (hl_labels_train['ltc2'][1], (hl_labels_train['ltc2'][1]*100.)/hl_num_train)
    print ""
    print "[Headlines]\t\t Dev +1 day BTC labeled 0: %d (%.2f%%)" % (hl_labels_dev['btc1'][0], (hl_labels_dev['btc1'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +1 day BTC labeled 1: %d (%.2f%%)" % (hl_labels_dev['btc1'][1], (hl_labels_dev['btc1'][1]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days BTC labeled 0: %d (%.2f%%)" % (hl_labels_dev['btc2'][0], (hl_labels_dev['btc2'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days BTC labeled 1: %d (%.2f%%)" % (hl_labels_dev['btc2'][1], (hl_labels_dev['btc2'][1]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +1 day ETH labeled 0: %d (%.2f%%)" % (hl_labels_dev['eth1'][0], (hl_labels_dev['eth1'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +1 day ETH labeled 1: %d (%.2f%%)" % (hl_labels_dev['eth1'][1], (hl_labels_dev['eth1'][1]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days ETH labeled 0: %d (%.2f%%)" % (hl_labels_dev['eth2'][0], (hl_labels_dev['eth2'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days ETH labeled 1: %d (%.2f%%)" % (hl_labels_dev['eth2'][1], (hl_labels_dev['eth2'][1]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +1 day LTC labeled 0: %d (%.2f%%)" % (hl_labels_dev['ltc1'][0], (hl_labels_dev['ltc1'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +1 day LTC labeled 1: %d (%.2f%%)" % (hl_labels_dev['ltc1'][1], (hl_labels_dev['ltc1'][1]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days LTC labeled 0: %d (%.2f%%)" % (hl_labels_dev['ltc2'][0], (hl_labels_dev['ltc2'][0]*100.)/hl_num_dev)
    print "[Headlines]\t\t Dev +2 days LTC labeled 1: %d (%.2f%%)" % (hl_labels_dev['ltc2'][1], (hl_labels_dev['ltc2'][1]*100.)/hl_num_train)
    print ""
    print "[Headlines]\t\t Test +1 day BTC labeled 0: %d (%.2f%%)" % (hl_labels_test['btc1'][0], (hl_labels_test['btc1'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +1 day BTC labeled 1: %d (%.2f%%)" % (hl_labels_test['btc1'][1], (hl_labels_test['btc1'][1]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days BTC labeled 0: %d (%.2f%%)" % (hl_labels_test['btc2'][0], (hl_labels_test['btc2'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days BTC labeled 1: %d (%.2f%%)" % (hl_labels_test['btc2'][1], (hl_labels_test['btc2'][1]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +1 day ETH labeled 0: %d (%.2f%%)" % (hl_labels_test['eth1'][0], (hl_labels_test['eth1'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +1 day ETH labeled 1: %d (%.2f%%)" % (hl_labels_test['eth1'][1], (hl_labels_test['eth1'][1]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days ETH labeled 0: %d (%.2f%%)" % (hl_labels_test['eth2'][0], (hl_labels_test['eth2'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days ETH labeled 1: %d (%.2f%%)" % (hl_labels_test['eth2'][1], (hl_labels_test['eth2'][1]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +1 day LTC labeled 0: %d (%.2f%%)" % (hl_labels_test['ltc1'][0], (hl_labels_test['ltc1'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +1 day LTC labeled 1: %d (%.2f%%)" % (hl_labels_test['ltc1'][1], (hl_labels_test['ltc1'][1]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days LTC labeled 0: %d (%.2f%%)" % (hl_labels_test['ltc2'][0], (hl_labels_test['ltc2'][0]*100.)/hl_num_test)
    print "[Headlines]\t\t Test +2 days LTC labeled 1: %d (%.2f%%)" % (hl_labels_test['ltc2'][1], (hl_labels_test['ltc2'][1]*100.)/hl_num_test)

#################################################################
# Print number of entries labeled '1' and '0' in tweet datasets
#################################################################
def print_labels_tweets(btc_labels, eth_labels, ltc_labels):
    # Split input
    btc_train, btc_dev, btc_test = btc_labels[0], btc_labels[1], btc_labels[2]
    eth_train, eth_dev, eth_test = eth_labels[0], eth_labels[1], eth_labels[2]
    ltc_train, ltc_dev, ltc_test = ltc_labels[0], ltc_labels[1], ltc_labels[2]
    # Get total number of entries in each dataset
    btc_num_train = btc_train['btc1'][0] + btc_train['btc1'][1]
    btc_num_dev = btc_dev['btc1'][0] + btc_dev['btc1'][1]
    btc_num_test = btc_test['btc1'][0] + btc_test['btc1'][1]
    eth_num_train = eth_train['eth1'][0] + eth_train['eth1'][1]
    eth_num_dev = eth_dev['eth1'][0] + eth_dev['eth1'][1]
    eth_num_test = eth_test['eth1'][0] + eth_test['eth1'][1]
    ltc_num_train = ltc_train['ltc1'][0] + ltc_train['ltc1'][1]
    ltc_num_dev = ltc_dev['ltc1'][0] + ltc_dev['ltc1'][1]
    ltc_num_test = ltc_test['ltc1'][0] + ltc_test['ltc1'][1]
    # Print
    print "\nLABEL COUNTS:"
    print "[Bitcoin tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (btc_train['btc1'][0], (btc_train['btc1'][0]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (btc_train['btc1'][1], (btc_train['btc1'][1]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (btc_train['btc2'][0], (btc_train['btc2'][0]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (btc_train['btc2'][1], (btc_train['btc2'][1]*100.)/btc_num_train)
    print ""
    print "[Bitcoin tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (btc_dev['btc1'][0], (btc_dev['btc1'][0]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (btc_dev['btc1'][1], (btc_dev['btc1'][1]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (btc_dev['btc2'][0], (btc_dev['btc2'][0]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (btc_dev['btc2'][1], (btc_dev['btc2'][1]*100.)/btc_num_dev)
    print ""
    print "[Bitcoin tweets]\t Test +1 day labeled 0: %d (%.2f%%)" % (btc_test['btc1'][0], (btc_test['btc1'][0]*100.)/btc_num_test)
    print "[Bitcoin tweets]\t Test +1 day labeled 1: %d (%.2f%%)" % (btc_test['btc1'][1], (btc_test['btc1'][1]*100.)/btc_num_test)
    print "[Bitcoin tweets]\t Test +2 days labeled 0: %d (%.2f%%)" % (btc_test['btc2'][0], (btc_test['btc2'][0]*100.)/btc_num_test)
    print "[Bitcoin tweets]\t Test +2 days labeled 1: %d (%.2f%%)" % (btc_test['btc2'][1], (btc_test['btc2'][1]*100.)/btc_num_test)
    print ""
    print "[Ethereum tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (eth_train['eth1'][0], (eth_train['eth1'][0]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (eth_train['eth1'][1], (eth_train['eth1'][1]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (eth_train['eth2'][0], (eth_train['eth2'][0]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (eth_train['eth2'][1], (eth_train['eth2'][1]*100.)/eth_num_train)
    print ""
    print "[Ethereum tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (eth_dev['eth1'][0], (eth_dev['eth1'][0]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (eth_dev['eth1'][1], (eth_dev['eth1'][1]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (eth_dev['eth2'][0], (eth_dev['eth2'][0]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (eth_dev['eth2'][1], (eth_dev['eth2'][1]*100.)/eth_num_dev)
    print ""
    print "[Ethereum tweets]\t Test +1 day labeled 0: %d (%.2f%%)" % (eth_test['eth1'][0], (eth_test['eth1'][0]*100.)/eth_num_test)
    print "[Ethereum tweets]\t Test +1 day labeled 1: %d (%.2f%%)" % (eth_test['eth1'][1], (eth_test['eth1'][1]*100.)/eth_num_test)
    print "[Ethereum tweets]\t Test +2 days labeled 0: %d (%.2f%%)" % (eth_test['eth2'][0], (eth_test['eth2'][0]*100.)/eth_num_test)
    print "[Ethereum tweets]\t Test +2 days labeled 1: %d (%.2f%%)" % (eth_test['eth2'][1], (eth_test['eth2'][1]*100.)/eth_num_test)
    print ""
    print "[Litecoin tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (ltc_train['ltc1'][0], (ltc_train['ltc1'][0]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (ltc_train['ltc1'][1], (ltc_train['ltc1'][1]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (ltc_train['ltc2'][0], (ltc_train['ltc2'][0]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (ltc_train['ltc2'][1], (ltc_train['ltc2'][1]*100.)/ltc_num_train)
    print ""
    print "[Litecoin tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (ltc_dev['ltc1'][0], (ltc_dev['ltc1'][0]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (ltc_dev['ltc1'][1], (ltc_dev['ltc1'][1]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (ltc_dev['ltc2'][0], (ltc_dev['ltc2'][0]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (ltc_dev['ltc2'][1], (ltc_dev['ltc2'][1]*100.)/ltc_num_dev)
    print ""
    print "[Litecoin tweets]\t Test +1 day labeled 0: %d (%.2f%%)" % (ltc_test['ltc1'][0], (ltc_test['ltc1'][0]*100.)/ltc_num_test)
    print "[Litecoin tweets]\t Test +1 day labeled 1: %d (%.2f%%)" % (ltc_test['ltc1'][1], (ltc_test['ltc1'][1]*100.)/ltc_num_test)
    print "[Litecoin tweets]\t Test +2 days labeled 0: %d (%.2f%%)" % (ltc_test['ltc2'][0], (ltc_test['ltc2'][0]*100.)/ltc_num_test)
    print "[Litecoin tweets]\t Test +2 days labeled 1: %d (%.2f%%)" % (ltc_test['ltc2'][1], (ltc_test['ltc2'][1]*100.)/ltc_num_test)

#################################################################
# Calculate the price changes based on coin data
#################################################################
def calc_percent_change(price_list, num_days):
    # Calculates in ascending order
    NUM_DAYS = len(price_list)
    # Create a numpy column vector to store price changes
    day_changes = np.zeros((NUM_DAYS, 1))
    # Calculate percent change
    for day in range(NUM_DAYS):
        if day < NUM_DAYS - num_days:
            day_changes[day + num_days] = ((price_list[day + num_days] - price_list[day]) / float(price_list[day])) * 100
    return day_changes

#################################################################
# Plot coin prices during dev/test time periods
#################################################################
def plot_coin_prices(all_prices):
    # Loop through each of our time periods (train, dev, and test)
    for period in TIME_PERIODS:
        # Setup
        plt.figure()
        plt.title('Coin Prices: %s to %s' % (period[0], period[1]))
        plt.style.use('bmh')
        colors = ['xkcd:blue', 'xkcd:indigo', 'xkcd:magenta']
        f, axs = plt.subplots(1, 3, figsize=(16,5))
        # Loop through each coin
        for i, coin in enumerate(COINS):
            # Select appropriate dataframe, then reverse order
            coin_df = all_prices[i]
            coin_df = coin_df.iloc[::-1]
            # Only keep prices during current time period
            coin_df = coin_df.loc[period[0]:period[1]]
            # Plot
            axs[i].plot(range(len(coin_df['Close'].values)), coin_df['Close'], label=coin, color=colors[i])
            axs[i].set_ylabel('Price in USD', size=18)
            axs[i].set_xlabel('Days', size=18)
            axs[i].legend()
        # Save
        plt.savefig('Coin Prices %s to %s.png' % (period[0], period[1]))

################################################################################################
################################################################################################
# Printing / plotting model results
################################################################################################
################################################################################################

#################################################################
# Print raw classifier results
#################################################################
def print_dev_results(classifier_results, description):  

    print "\n%s:" % description
    print "TRAINING:"
    print "Accuracy:\t %.2f" % (classifier_results['train_accuracy'] * 100)
    print "TESTING:"
    print "Accuracy:\t %.2f" % (classifier_results['accuracy'] * 100)
    print "Precision:\t %.2f" % (classifier_results['precision'] * 100)
    print "Recall:\t\t %.2f" % (classifier_results['recall'] * 100)
    print "F1-score:\t %.2f" % (classifier_results['F1'] * 100)
    print "Confusion Matrix:\n%s" % classifier_results['confusion_matrix']
    print "Normalized Confusion Matrix:\n%s" % classifier_results['normalized_confusion_matrix']

#################################################################
# Print info on final model results
#################################################################
def print_predictions(data_type, classifier, results):
    # Describe model run
    print "\n%s | %s:" % (data_type, classifier)
    # Overall accuracy
    print "--------------------------------"
    print "--------------------------------"
    print "OVERALL:"
    print "BTC1: %.2f%%, BTC2: %.2f%%, ETH1: %.2f%%, ETH2: %.2f%%, LTC1: %.2f%%, LTC2: %.2f%%" % \
        ((results[0]['cache']['accuracy'] * 100), (results[1]['cache']['accuracy'] * 100), 
        (results[2]['cache']['accuracy'] * 100), (results[3]['cache']['accuracy'] * 100), 
        (results[4]['cache']['accuracy'] * 100), (results[5]['cache']['accuracy'] * 100))
    # Loop through each of the coins (only care about 1 day price predictions)
    i = -2
    for coin in COINS:
        i += 2
        print "--------------------------------"
        print "--------------------------------"
        print "%s:" % coin.upper()
        # Correct price increase predictions
        print "--------------------------------"
        print "Price INCREASE (1 Day), Prediction CORRECT"
        price_increase_1day_df = pd.Series(results[i]['cache']['price_increase'])
        print price_increase_1day_df.describe()
        # Incorrect price increase predictions
        print "--------------------------------"
        print "Price INCREASE (1 Day), Prediction INCORRECT"
        price_increase_1day_wrong_df = pd.Series(results[i]['cache']['price_increase_wrong'])
        print price_increase_1day_wrong_df.describe()
        # Correct price decrease predictions
        print "--------------------------------"
        print "Price DECREASE (1 Day), Prediction CORRECT"
        price_decrease_1day_df = pd.Series(results[i]['cache']['price_decrease'])
        print price_decrease_1day_df.describe()
        # Incorrect price decrease predictions
        print "--------------------------------"
        print "Price DECREASE (1 Day), Prediction INCORRECT"
        price_decrease_1day_wrong_df = pd.Series(results[i]['cache']['price_decrease_wrong'])
        print price_decrease_1day_wrong_df.describe()
        # Stats for the full DataFrame to see how we compare to the average increases/decreases and max/mins
        print "--------------------------------"
        print results[i]['pred_df']['1D Percent Change'].describe()

#################################################################
# Plot predicted price changes
#################################################################
def plot_predictions(data_type, classifier, results):
    # Loop through each of the coins (only care about 1 day price predictions)
    i = -2
    for coin in COINS:
        i += 2    
        # Setup
        plt.figure(figsize=(14, 6))
        plt.style.use('bmh')
        plt.title("%s - %s - 1 Day Price Predictions" % (data_type, coin))
        colors = ['xkcd:crimson', 'xkcd:darkgreen']
        # Determine % changes and corresponding plot colors
        pred_colors = []
        percent_changes = []
        changes = results[i]['changes']
        for change in changes:
            price_change, pred = change
            if price_change > 0:
                if pred == 1:  # correct
                    percent_changes.append(price_change)
                    pred_colors.append(colors[1])
                else:  # incorrect
                    percent_changes.append(price_change)
                    pred_colors.append(colors[0])
            else:
                if pred == -1:  # correct
                    percent_changes.append(price_change)
                    pred_colors.append(colors[1])
                else:
                    percent_changes.append(price_change)
                    pred_colors.append(colors[0])
        # Plot
        plt.bar(range(len(percent_changes)), percent_changes, color=pred_colors)
        plt.ylabel('Coin Price % Change', size=18)
        plt.xlabel('Days', size=18)
        # Save
        plt.savefig("%s-%s-1DayPricePredictions.png" % (data_type, coin))

#################################################################
# Plot top features (words) and their learned weights
#################################################################
def plot_top_weights(description, classifier, feature_names, top_features):

    # Determine top positive and negative weights
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # Plot
    plt.figure(figsize=(30, 15))
    plt.title(description)
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig("%s - Top Feature Weights.png" % description)

#################################################################
# Plot multiple sets of top features and their weights
#################################################################
def plot_multi_top_weights(data_type, results, top_features):

    # Repeatedly call plot_top_weights for each type of prediction label
    descriptions = ['%s - BTC 1 Day' % data_type, '%s - BTC 2 Days' % data_type, 
                    '%s - ETH 1 Day' % data_type, '%s - ETH 2 Days' % data_type, 
                    '%s - LTC 1 Day' % data_type, '%s - LTC 2 Days' % data_type]
    i = -1
    for item in descriptions:
        i += 1
        plot_top_weights(item, results[i]['classifier'], results[i]['feature-names'], top_features)