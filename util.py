import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import string

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score

#################################################################
# Setup constant values to be used later 
#################################################################

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

# Train, dev, and test dates
TRAIN_START_DATE = '2017-01-01'
DEV_START_DATE = '2017-7-20'
DEV_END_DATE = '2017-09-23'
TEST_START_DATE = '2017-09-24'
TEST_END_DATE = '2017-11-30'

# Parser and punctuation variables
parser = spacy.load('en_core_web_sm')
punctuations = string.punctuation

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
# Print number of entries in dataset
#################################################################
def print_num_entries_headlines(hl_num_train, hl_num_dev):
    print "\nTOTAL ENTRY COUNTS:"
    print "[Headlines]\t\t Total train entries: %d, Total dev entries: %d" % (hl_num_train, hl_num_dev)
def print_num_entries_tweets(btc_num_train, btc_num_dev, eth_num_train, eth_num_dev, ltc_num_train, ltc_num_dev):
    print "\nTOTAL ENTRY COUNTS:"
    print "[Bitcoin tweets]\t Total train entries: %d, Total dev entries: %d" % (btc_num_train, btc_num_dev)
    print "[Ethereum tweets]\t Total train entries: %d, Total dev entries: %d" % (eth_num_train, eth_num_dev)
    print "[Litcoin tweets]\t Total train entries: %d, Total dev entries: %d" % (ltc_num_train, ltc_num_dev)

#################################################################
# Function to calculate the price changes based on coin data
#################################################################
def calc_percent_change(price_list, num_days):
    # Calculates in ascending order
    NUM_DAYS = len(price_list)
    # Create a numpy column vector to store price changes
    day_changes = np.zeros((NUM_DAYS, 1))
    # Calculate percent change
    for day in range(NUM_DAYS):
        if day < NUM_DAYS - num_days:
            day_changes[day + num_days] = ((price_list[day + num_days] - price_list[day]) \
                                           / float(price_list[day])) * 100
    return day_changes

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

    return toReturn

#################################################################
# Print number of entries labeled '1' and '0' in dataset
#################################################################
def print_labels_headlines(hl_labels_train, hl_labels_dev):

    # Get total number of entries in each dataset
    hl_num_train = hl_labels_train['btc1'][0] + hl_labels_train['btc1'][1]
    hl_num_dev = hl_labels_dev['btc1'][0] + hl_labels_dev['btc1'][1]

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

def print_labels_tweets(btc_labels_train, btc_labels_dev, eth_labels_train, eth_labels_dev, ltc_labels_train, ltc_labels_dev):

    # Get total number of entries in each dataset
    btc_num_train = btc_labels_train['btc1'][0] + btc_labels_train['btc1'][1]
    btc_num_dev = btc_labels_dev['btc1'][0] + btc_labels_dev['btc1'][1]
    eth_num_train = eth_labels_train['eth1'][0] + eth_labels_train['eth1'][1]
    eth_num_dev = eth_labels_dev['eth1'][0] + eth_labels_dev['eth1'][1]
    ltc_num_train = ltc_labels_train['ltc1'][0] + ltc_labels_train['ltc1'][1]
    ltc_num_dev = ltc_labels_dev['ltc1'][0] + ltc_labels_dev['ltc1'][1]

    # Print
    print "\nLABEL COUNTS:"
    print "[Bitcoin tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (btc_labels_train['btc1'][0], (btc_labels_train['btc1'][0]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (btc_labels_train['btc1'][1], (btc_labels_train['btc1'][1]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (btc_labels_train['btc2'][0], (btc_labels_train['btc2'][0]*100.)/btc_num_train)
    print "[Bitcoin tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (btc_labels_train['btc2'][1], (btc_labels_train['btc2'][1]*100.)/btc_num_train)
    print ""
    print "[Bitcoin tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (btc_labels_dev['btc1'][0], (btc_labels_dev['btc1'][0]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (btc_labels_dev['btc1'][1], (btc_labels_dev['btc1'][1]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (btc_labels_dev['btc2'][0], (btc_labels_dev['btc2'][0]*100.)/btc_num_dev)
    print "[Bitcoin tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (btc_labels_dev['btc2'][1], (btc_labels_dev['btc2'][1]*100.)/btc_num_dev)
    print ""
    print "[Ethereum tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (eth_labels_train['eth1'][0], (eth_labels_train['eth1'][0]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (eth_labels_train['eth1'][1], (eth_labels_train['eth1'][1]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (eth_labels_train['eth2'][0], (eth_labels_train['eth2'][0]*100.)/eth_num_train)
    print "[Ethereum tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (eth_labels_train['eth2'][1], (eth_labels_train['eth2'][1]*100.)/eth_num_train)
    print ""
    print "[Ethereum tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (eth_labels_dev['eth1'][0], (eth_labels_dev['eth1'][0]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (eth_labels_dev['eth1'][1], (eth_labels_dev['eth1'][1]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (eth_labels_dev['eth2'][0], (eth_labels_dev['eth2'][0]*100.)/eth_num_dev)
    print "[Ethereum tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (eth_labels_dev['eth2'][1], (eth_labels_dev['eth2'][1]*100.)/eth_num_dev)
    print ""
    print "[Litecoin tweets]\t Train +1 day labeled 0: %d (%.2f%%)" % (ltc_labels_train['ltc1'][0], (ltc_labels_train['ltc1'][0]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +1 day labeled 1: %d (%.2f%%)" % (ltc_labels_train['ltc1'][1], (ltc_labels_train['ltc1'][1]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +2 days labeled 0: %d (%.2f%%)" % (ltc_labels_train['ltc2'][0], (ltc_labels_train['ltc2'][0]*100.)/ltc_num_train)
    print "[Litecoin tweets]\t Train +2 days labeled 1: %d (%.2f%%)" % (ltc_labels_train['ltc2'][1], (ltc_labels_train['ltc2'][1]*100.)/ltc_num_train)
    print ""
    print "[Litecoin tweets]\t Dev +1 day labeled 0: %d (%.2f%%)" % (ltc_labels_dev['ltc1'][0], (ltc_labels_dev['ltc1'][0]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +1 day labeled 1: %d (%.2f%%)" % (ltc_labels_dev['ltc1'][1], (ltc_labels_dev['ltc1'][1]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +2 days labeled 0: %d (%.2f%%)" % (ltc_labels_dev['ltc2'][0], (ltc_labels_dev['ltc2'][0]*100.)/ltc_num_dev)
    print "[Litecoin tweets]\t Dev +2 days labeled 1: %d (%.2f%%)" % (ltc_labels_dev['ltc2'][1], (ltc_labels_dev['ltc2'][1]*100.)/ltc_num_dev)

#################################################################
# Custom transformer using spaCy 
#################################################################
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

#################################################################
# Basic utility function to clean the text 
#################################################################
def clean_text(text):     
    return text.strip().lower()

#################################################################
# Create spacy tokenizer that parses a sentence and generates tokens
# these can also be replaced by word vectors (i.e. word2vec or glove)
#################################################################
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    #tokens = [tok for tok in tokens if not (tok.pos_ == "NUM" and not tok.is_alpha)]  # Uncomment this line to remove numbers
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)] 
    return tokens

#################################################################
# Train and evaluate dev predictions on specified dataset / labels
#################################################################
def train_and_dev(train_df, dev_df, data_type, label_column, classifier_type):

    # Create vectorizer object to generate feature vectors, we will use the custom tokenizer
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    # Choose classifier
    if classifier_type == 'linear_svc':
        classifier = LinearSVC()
    elif classifier_type == 'multinomial_nb':
        classifier = MultinomialNB()
    elif classifier_type == 'bernoulli_nb':
        classifier = BernoulliNB() # This seems promising
    else: # classifier_type == 'logistic_regression':
        classifier = LogisticRegression()

    # Create the  pipeline to clean, tokenize, vectorize, and classify 
    pipe = Pipeline([('cleaner', predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

    # Choose text column
    if data_type == 'headlines':
        text_column = HEADLINE_TEXT_COLUMN
    else:
        text_column = TWEET_TEXT_COLUMN

    # Get the text and labels form the dataframe as numpy arrays
    train_x, train_y, train_date = train_df[text_column].values, train_df[label_column].values, train_df[DATE].values
    dev_x, dev_y, dev_date = dev_df[text_column].values, dev_df[label_column].values, dev_df[DATE].values

    # Create model and measure accuracy on validation set (not the test set)
    pipe.fit(train_x, train_y)
    pred_data = pipe.predict(dev_x)
	
    #Make single prediction based on all labels on a given day	
    daily_predictions = {date:[0, 0] for date in dev_date} #date: [predicted, actual]
    for i in range(len(dev_date)):
	
        #add (predicted label - 0.5) so we can just check the sign of the result to get the prediction
        daily_predictions[dev_date[i]][0] += float(pred_data[i]-0.5)
        daily_predictions[dev_date[i]][1] += float(dev_y[i]-0.5)
		
    #count the results of predictions 
    predUp_yUp = 0 #predict up and actual goes up
    predDown_yUp = 0 #predict down and actual goes up
    predUp_yDown = 0 #predict up and actual goes down
    predDown_yDown = 0 #predict down and actual goes down

    for dp in daily_predictions: #currently ignoring the possibility of a prediction being split between positive and negative
        if daily_predictions[dp][0] > 0:
            if daily_predictions[dp][1] > 0:
                predUp_yUp += 1
            else:
                predUp_yDown += 1
        else:
            if daily_predictions[dp][1] > 0:
                predDown_yUp += 1
            else:
                predDown_yDown += 1
    
    #new metric and a baseline metric
    daily_accuracy = (predUp_yUp+predDown_yDown)/float(len(daily_predictions)) 
    alwaysPredictUp_accuracy = (predUp_yUp+predDown_yUp)/float(len(daily_predictions))
    				
    cm = confusion_matrix(dev_y, pred_data)

    # Return
    toReturn = {}
    toReturn['accuracy'] = accuracy_score(dev_y, pred_data)
    toReturn['precision'] = precision_score(dev_y, pred_data)
    toReturn['recall'] = recall_score(dev_y, pred_data)
    toReturn['F1'] = f1_score(dev_y, pred_data)
    toReturn['confusion-matrix'] = cm
    toReturn['normalized-confusion-matrix'] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    toReturn['daily_accuracy'] = daily_accuracy

    toReturn['classifier'] = classifier
    toReturn['feature-names'] = vectorizer.get_feature_names()

    return toReturn

#################################################################
# Plot top features (words) and their learned weights
#################################################################
def see_top_weights(description, classifier, feature_names, top_features=30):

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
    plt.savefig(description + ".png")

#################################################################
# Plot predicted price changes
#################################################################
def plot_predicted_changes(changes, coin_name, media_name):
        colors = ['xkcd:crimson', 'xkcd:darkgreen']
        pred_colors = []
        percent_changes = []
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

        plt.style.use('bmh')
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(percent_changes)), percent_changes, color=pred_colors)
        plt.ylabel('Coin Price % Change', size=18)
        plt.xlabel('Days', size=18)
        plt.show()

#################################################################
# Model evaluation 
#################################################################
def get_increase_decrease(df, sum_pred_labels_column, change_column, num_days_change, counts_per_day_column):
    
    #df: (DataFrame) that contains the counts of labels per day and price change
    #sum_pred_labels_column: (string) Column of predicted labels corresponding to unit of change used
    #change_column: (string) The name of the column containing the amount of change
    #num_days_of_change: (int) The number of days in the past change was calculated for
    #counts_per_day_column: (string) The name of the column that has the counts of tweets/headlines per day
    
    # Note: This algorithm assumes that the labes were changed from 1/0 to 1/-1 after prediction so that when summed
    #       we get a value that represents the total number of both positive and negative labels

    # 1) Extract the necessary columns as numpy arrays
    counts = df[counts_per_day_column].values
    sum_pred_labels = df[sum_pred_labels_column].values
    changes = df[change_column].values
    num_total_days = df.shape[0]

    # 2) perform algorithm
    cache = {}
    cache['price_increase'] = []
    cache['price_decrease'] = []
    cache['price_increase_wrong'] = []
    cache['price_decrease_wrong'] = []
    cache['all_price_increase'] = []
    cache['all_price_decrease'] = []
    all_changes = []

    for day in range(num_total_days - num_days_change):
        price_change = changes[day + num_days_change]
        # if the sum of predicted labels is the majority and the actual price increased, add to list
        if price_change > 0:
            if sum_pred_labels[day] > float(counts[day]) / 2.:
                cache['price_increase'].append(price_change)
                cache['all_price_increase'].append((price_change, 1))
                all_changes.append((price_change, 1))
            else:
                cache['price_increase_wrong'].append(price_change)
                cache['all_price_increase'].append((price_change, 0))
                all_changes.append((price_change, 2))

        # if the sum of predicted labels is not the majority and the actual price decreased
        elif price_change < 0:
            if sum_pred_labels[day] < abs(float(counts[day]) / 2.):
                cache['price_decrease'].append(price_change)
                cache['all_price_decrease'].append((price_change, 1))
                all_changes.append((price_change, -1))
            else:
                cache['price_decrease_wrong'].append(price_change)
                cache['all_price_decrease'].append((price_change, 0))
                all_changes.append((price_change, -2))

    return cache, all_changes


#################################################################
# Print dev prediction results
#################################################################
def print_dev_results(dev_results, description):  

    print "\n%s:" % description
    print "Accuracy:\t %.2f" % (dev_results['accuracy'] * 100)
    print "Precision:\t %.2f" % (dev_results['precision'] * 100)
    print "Recall:\t\t %.2f" % (dev_results['recall'] * 100)
    print "F1-score:\t %.2f" % (dev_results['F1'] * 100)
    print "Confusion Matrix:\n%s" % dev_results['confusion-matrix']
    print "Normalized Confusion Matrix:\n%s" % dev_results['normalized-confusion-matrix']

############################################################################
# Main
############################################################################
def main():

if __name__ == '__main__':
	main()
