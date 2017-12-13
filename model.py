import pandas as pd
import numpy as np

from util import *

################################################################################################
################################################################################################
# Setup constant values to be used later 
################################################################################################
################################################################################################

# Labeled data file paths (news headlines and tweets)
HEADLINES_TRAIN_FILE = "data/news_and_tweets/labeled/headlines_labeled_train.csv"
HEADLINES_DEV_FILE = "data/news_and_tweets/labeled/headlines_labeled_dev.csv"
HEADLINES_TEST_FILE = "data/news_and_tweets/labeled/headlines_labeled_test.csv"
BTC_TRAIN_FILE = "data/news_and_tweets/labeled/bitcoin_tweets_labeled_train.csv"
BTC_DEV_FILE = "data/news_and_tweets/labeled/bitcoin_tweets_labeled_dev.csv"
BTC_TEST_FILE = "data/news_and_tweets/labeled/bitcoin_tweets_labeled_test.csv"
ETH_TRAIN_FILE = "data/news_and_tweets/labeled/ethereum_tweets_labeled_train.csv"
ETH_DEV_FILE = "data/news_and_tweets/labeled/ethereum_tweets_labeled_dev.csv"
ETH_TEST_FILE = "data/news_and_tweets/labeled/ethereum_tweets_labeled_test.csv"
LTC_TRAIN_FILE = "data/news_and_tweets/labeled/litecoin_tweets_labeled_train.csv"
LTC_DEV_FILE = "data/news_and_tweets/labeled/litecoin_tweets_labeled_dev.csv"
LTC_TEST_FILE = "data/news_and_tweets/labeled/litecoin_tweets_labeled_test.csv"

# Labeled data file paths (coin prices)
BTC_PRICE_FILE = "data/coin_prices/bitcoin_price.csv"
ETH_PRICE_FILE = "data/coin_prices/ethereum_price.csv"
LTC_PRICE_FILE = "data/coin_prices/litecoin_price.csv"

# Column names (for pulling data out of csv files, making predictions)
HEADLINE_TEXT_COLUMN = 'headline'
TWEET_TEXT_COLUMN = 'tweet'
BTC_COL_1_D, BTC_COL_2_D = 'bitcoin_one', 'bitcoin_two'
ETH_COL_1_D, ETH_COL_2_D = 'ethereum_one', 'ethereum_two'
LTC_COL_1_D, LTC_COL_2_D = 'litecoin_one', 'litecoin_two'
PREDICTION_LABELS = [BTC_COL_1_D, BTC_COL_2_D, ETH_COL_1_D, ETH_COL_2_D, LTC_COL_1_D, LTC_COL_2_D]

# Classifiers
CLASSIFIERS = ['logistic_regression', 'linear_svc', 'multinomial_nb', 'bernoulli_nb']

################################################################################################
################################################################################################
# Machine learning pipeline functions
################################################################################################
################################################################################################

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
    return str(text).strip().lower()

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
# Train model and predict data labels for specified dataset / labels
#################################################################
def model_fit_and_predict(run_type, data_type, media_data, label_column, classifier_type):

    # Separate media_data
    train_df, dev_df, test_df = media_data[0], media_data[1], media_data[2]

    # Create vectorizer object to generate feature vectors, we will use the custom tokenizer
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,2))

    # Choose classifier
    if classifier_type == 'linear_svc':
        classifier = LinearSVC()
    elif classifier_type == 'multinomial_nb':
        classifier = MultinomialNB()
    elif classifier_type == 'bernoulli_nb':
        classifier = BernoulliNB()
    else: # classifier_type == 'logistic_regression':
        classifier = LogisticRegression(C=0.9)

    # Create the  pipeline to clean, tokenize, vectorize, and classify 
    pipe = Pipeline([('cleaner', predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

    # Choose text column
    if data_type == 'headlines':
        text_column = HEADLINE_TEXT_COLUMN
    else:
        text_column = TWEET_TEXT_COLUMN

    # If running for model validation, fit on train set and predict on dev set
    if (run_type == 'validation'):
        fit_x, fit_y = train_df[text_column].values, train_df[label_column].values
        pred_x, pred_y = dev_df[text_column].values, dev_df[label_column].values

    # Else if running model for final testing, fit on train+dev sets and predict on test set
    else:       
        train_df = train_df.append(dev_df)
        fit_x, fit_y = train_df[text_column].values, train_df[label_column].values
        pred_x, pred_y = test_df[text_column].values, test_df[label_column].values

    # Train model, save training accuracy
    pipe.fit(fit_x, fit_y)
    train_accuracy = pipe.score(fit_x, fit_y)

    # Use trained model to predict labels
    pred_results = pipe.predict(pred_x)
    				
    # Return
    toReturn = {}
    toReturn['train_accuracy'] = train_accuracy
    toReturn['accuracy'] = accuracy_score(pred_y, pred_results)
    toReturn['precision'] = precision_score(pred_y, pred_results)
    toReturn['recall'] = recall_score(pred_y, pred_results)
    toReturn['F1'] = f1_score(pred_y, pred_results)
    cm = confusion_matrix(pred_y, pred_results)
    toReturn['confusion_matrix'] = cm
    toReturn['normalized_confusion_matrix'] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    toReturn['pred_results'] = pred_results
    toReturn['classifier'] = classifier
    toReturn['feature-names'] = vectorizer.get_feature_names()
    return toReturn

#################################################################
# Aggregate model predictions by date and use to predict coin price changes
#################################################################
def predict_prices(run_type, data_type, media_data, all_prices, label_column, classifier_results):

    # Separate media_data
    train_df, dev_df, test_df = media_data[0], media_data[1], media_data[2]

    # Select appropriate fit date (always train set)
    fit_start_date = TRAIN_START_DATE

    # Select appropriate predict dates / data (either dev or test set)
    if (run_type == 'validation'):
        pred_start_date = DEV_START_DATE
        pred_end_date = DEV_END_DATE
        pred_df = dev_df.copy()
    else:
        pred_start_date = TEST_START_DATE
        pred_end_date = TEST_END_DATE 
        pred_df = test_df.copy()

    # Select appropriate coin price dataframe
    if (label_column == BTC_COL_1_D or label_column == BTC_COL_2_D):
        which_coin = 'bitcoin'
        coin_df = all_prices[0]
    elif (label_column == ETH_COL_1_D or label_column == ETH_COL_2_D):
        which_coin = 'ethereum'
        coin_df = all_prices[1]
    else:
        which_coin = 'litecoin'
        coin_df = all_prices[2]

    # Select appropriate text column
    if data_type == 'headlines':
        text_column = HEADLINE_TEXT_COLUMN
    else:
        text_column = TWEET_TEXT_COLUMN

    # Reverse date order, remove all prices before training start date
    coin_df = coin_df.iloc[::-1]
    coin_df = coin_df.loc[fit_start_date:]

    # Calculate 1 day price changes
    one_day_changes = calc_percent_change(coin_df['Close'].values, num_days=1)
    coin_df['1D Percent Change'] = one_day_changes

    # Add the predicted labels to the dataframe
    predicted_labels_column = '%s_pred' % which_coin
    pred_df[predicted_labels_column] = classifier_results

    # Replace the 0's with -1's so when we sum() it will be easy
    pred_df = pred_df.replace(0, -1)

    # Access only the dates in the test set range, then sum all the labels for each date
    pred_df = pred_df.drop(text_column, axis=1)
    pred_df = pred_df.loc[pred_start_date:pred_end_date].groupby(['date']).transform(sum)

    # Need this secondary DataFrame to get counts
    counts_df = pd.DataFrame(pred_df.groupby(['date']).size(), columns=['counts'])

    # Cleanup the DataFrame
    pred_df = pred_df.drop_duplicates()

    # Merge all the stuff we just created into one DataFrame
    pred_df = pd.concat([pred_df, coin_df, counts_df], axis=1, join_axes=[pred_df.index])

    # Finally, evaluate the price predictions (for 1 day out)
    days_change = 1
    cache, changes = get_increase_decrease(pred_df, predicted_labels_column, '1D Percent Change', days_change, 'counts')

    # Return
    toReturn = {}
    toReturn['cache'] = cache
    toReturn['changes'] = changes
    toReturn['pred_df'] = pred_df
    return toReturn

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

    # 2) Perform algorithm
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
        # If ACTUAL price increase 
        if price_change > 0:
            # If PREDICTED correctly
            if sum_pred_labels[day] > float(counts[day]) / 2.:
                cache['price_increase'].append(price_change)
                cache['all_price_increase'].append((price_change, 1))
                all_changes.append((price_change, 1))
            # Else PREDICTED incorrectly
            else:
                cache['price_increase_wrong'].append(price_change)
                cache['all_price_increase'].append((price_change, 0))
                all_changes.append((price_change, 2))

        # If ACTUAL price decrease
        elif price_change < 0:
            # If PREDICTED correctly
            if sum_pred_labels[day] < abs(float(counts[day]) / 2.):
                cache['price_decrease'].append(price_change)
                cache['all_price_decrease'].append((price_change, 1))
                all_changes.append((price_change, -1))
            # Else PREDICTED incorrectly
            else:
                cache['price_decrease_wrong'].append(price_change)
                cache['all_price_decrease'].append((price_change, 0))
                all_changes.append((price_change, -2))

    # Calculate total accuracy (# correct predictions / # total predictions)
    num_correct = len(cache['price_increase']) + len(cache['price_decrease'])
    num_total = len(cache['all_price_increase']) + len(cache['all_price_decrease'])
    cache['accuracy'] = float(num_correct) / float(num_total)

    # Return
    return cache, all_changes

################################################################################################
################################################################################################
# Run model (the function that ties everything together)
################################################################################################
################################################################################################
def run_model(run_type, classifier, hl_all_data, tweets_all_data, all_prices, print_results=True, plot_results=True):

    # Split twitter input data
    btc_all_data, eth_all_data, ltc_all_data = tweets_all_data[0], tweets_all_data[1], tweets_all_data[2]

    #################################
    # Headlines
    #################################
    if False:
        print "Predicting based on news headlines..."
        hl_classifier_results = []
        hl_final_predictions = []
        # Loop through all prediction labels
        for label in PREDICTION_LABELS:
            # Run model to assign labels to dev/test data, then use to make coin price predictions
            classifier_results = model_fit_and_predict(run_type, 'headlines', hl_all_data, label, classifier)
            final_predictions = predict_prices(run_type, 'headlines', hl_all_data, all_prices, label, classifier_results['pred_results'])
            # Save results
            hl_classifier_results.append(classifier_results)
            hl_final_predictions.append(final_predictions)
            print "Finished %s." % label
        # Print results
        if (print_results): print_predictions('Headlines', classifier, hl_final_predictions)
        # Plot results
        if (plot_results): plot_predictions('Headlines', classifier, hl_final_predictions)

    #################################
    # Tweets
    #################################
    print "Predicting based on tweets..."
    t_classifier_results = []
    t_final_predictions = []
    # Loop through all prediction labels
    for label in PREDICTION_LABELS:
        # Select tweet data appropriately for label
        if (label == BTC_COL_1_D or label == BTC_COL_2_D):  # Bitcoin
            media_data = btc_all_data
        elif (label == ETH_COL_1_D or label == ETH_COL_2_D): # Ethereum
            media_data = eth_all_data
        else: # Litecoin
            media_data = ltc_all_data
        # Run model to assign labels to dev/test data, then use to make coin price predictions
        classifier_results = model_fit_and_predict(run_type, 'tweets', media_data, label, classifier)
        final_predictions = predict_prices(run_type, 'tweets', media_data, all_prices, label, classifier_results['pred_results'])
        # Save results
        t_classifier_results.append(classifier_results)
        t_final_predictions.append(final_predictions)
        print "Finished %s." % label
    # Print results
    if (print_results): print_predictions('Tweets', classifier, t_final_predictions)
    # Plot results
    if (plot_results): plot_predictions('Tweets', classifier, t_final_predictions) 

################################################################################################
################################################################################################
# Main
################################################################################################
################################################################################################
def main():

    #######################################################
    # Setup
    #######################################################
    print "Loading data..."

    # Load headlines
    hl_train = load_csv(HEADLINES_TRAIN_FILE, HEADLINE_TEXT_COLUMN)
    hl_dev = load_csv(HEADLINES_DEV_FILE, HEADLINE_TEXT_COLUMN)
    hl_test = load_csv(HEADLINES_TEST_FILE, HEADLINE_TEXT_COLUMN)
    hl_all_data = [hl_train, hl_dev, hl_test]

    # Load tweets
    btc_tweets_train = load_csv(BTC_TRAIN_FILE, TWEET_TEXT_COLUMN)
    btc_tweets_dev = load_csv(BTC_DEV_FILE, TWEET_TEXT_COLUMN)
    btc_tweets_test = load_csv(BTC_TEST_FILE, TWEET_TEXT_COLUMN)
    eth_tweets_train = load_csv(ETH_TRAIN_FILE, TWEET_TEXT_COLUMN)
    eth_tweets_dev = load_csv(ETH_DEV_FILE, TWEET_TEXT_COLUMN)
    eth_tweets_test = load_csv(ETH_TEST_FILE, TWEET_TEXT_COLUMN)
    ltc_tweets_train = load_csv(LTC_TRAIN_FILE, TWEET_TEXT_COLUMN)
    ltc_tweets_dev = load_csv(LTC_DEV_FILE, TWEET_TEXT_COLUMN)
    ltc_tweets_test = load_csv(LTC_TEST_FILE, TWEET_TEXT_COLUMN)
    btc_all_data = [btc_tweets_train, btc_tweets_dev, btc_tweets_test] 
    eth_all_data = [eth_tweets_train, eth_tweets_dev, eth_tweets_test]
    ltc_all_data = [ltc_tweets_train, ltc_tweets_dev, ltc_tweets_test]
    tweets_all_data = [btc_all_data, eth_all_data, ltc_all_data]

    # Load coin prices
    btc_prices = pd.read_csv(BTC_PRICE_FILE, index_col=['Date'], parse_dates=True, usecols=['Date', 'Close', 'Volume', 'Market Cap'])
    eth_prices = pd.read_csv(ETH_PRICE_FILE, index_col=['Date'], parse_dates=True, usecols=['Date', 'Close', 'Volume', 'Market Cap'])
    ltc_prices = pd.read_csv(LTC_PRICE_FILE, index_col=['Date'], parse_dates=True, usecols=['Date', 'Close', 'Volume', 'Market Cap'])
    all_prices = [btc_prices, eth_prices, ltc_prices]

    #######################################################
    # Default (run model using final hyperparameters)
    #######################################################
    if (False):
        print "Running model using default configurations..."
        run_model('test', 'logistic_regression', hl_all_data, tweets_all_data, all_prices, print_results=True, plot_results=True)

    #######################################################
    # Experiment 0.1: Plot coin prices during train, dev, and test time periods
    #######################################################
    if (False):
        plot_coin_prices(all_prices)

    #######################################################
    # Experiment 0.2: Get info about input data
    #######################################################
    if (False):

        #################################
        # Headlines
        #################################
        # Number
        hl_counts = []
        for dataset in hl_all_data:
            hl_counts.append(count_num_entries(dataset))
        print_num_entries_headlines(hl_counts)
        # Label distribution
        hl_labels = []
        for dataset in hl_all_data:
            hl_labels.append(count_labels(dataset, 'headlines'))
        print_labels_headlines(hl_labels)

        #################################
        # Tweets
        #################################
        # Number
        btc_counts = []
        eth_counts = []
        ltc_counts = []
        for dataset in btc_all_data:
            btc_counts.append(count_num_entries(dataset))
        for dataset in eth_all_data:
            eth_counts.append(count_num_entries(dataset))
        for dataset in ltc_all_data:
            ltc_counts.append(count_num_entries(dataset))
        print_num_entries_tweets(btc_counts, eth_counts, ltc_counts)
        # Label distribution
        btc_labels = []
        eth_labels = []
        ltc_labels = []
        for dataset in btc_all_data:
            btc_labels.append(count_labels(dataset, 'btc_tweets'))
        for dataset in eth_all_data:
            eth_labels.append(count_labels(dataset, 'eth_tweets'))
        for dataset in ltc_all_data:
            ltc_labels.append(count_labels(dataset, 'ltc_tweets'))       
        print_labels_tweets(btc_labels, eth_labels, ltc_labels)

    #######################################################
    # Experiment 1: Try different types of classifiers
    #######################################################
    if (False):

        # Choose if running for model validation (dev set) or final testing (test set)
        run_type = 'validation'
        #run_type = 'test'

        # Loop through all classifiers
        for classifier in CLASSIFIERS:
            print "Running model using %s classifier..." % classifier
            run_model(run_type, classifier, hl_all_data, tweets_all_data, all_prices, print_results=True, plot_results=False)

    #######################################################
    # Experiment 2: Looking at feature weights
    #######################################################
    if (True):

        # Choose if running for model validation (dev set) or final testing (test set)
        run_type = 'validation'
        #run_type = 'test'

        # Choose number of top features to see weights for
        num_top_features = 30

        # Headlines
        print "Predicting based on news headlines..."
        hl_classifier_results = []
        # Loop through all prediction labels
        for label in PREDICTION_LABELS:
            # Run model to assign labels to dev/test data, save results
            classifier_results = model_fit_and_predict(run_type, 'headlines', hl_all_data, label, 'logistic_regression')
            hl_classifier_results.append(classifier_results)
            print "Finished %s." % label
        # Plot results
        plot_multi_top_weights('Headlines', hl_classifier_results, num_top_features)

        # Tweets
        print "Predicting based on tweets..."
        t_classifier_results = []
        # Loop through all prediction labels
        for label in PREDICTION_LABELS:
            # Select tweet data appropriately for label
            if (label == BTC_COL_1_D or label == BTC_COL_2_D):  # Bitcoin
                media_data = btc_all_data
            elif (label == ETH_COL_1_D or label == ETH_COL_2_D): # Ethereum
                media_data = eth_all_data
            else: # Litecoin
                media_data = ltc_all_data
            # Run model to assign labels to dev/test data, save results
            classifier_results = model_fit_and_predict(run_type, 'tweets', media_data, label, 'logistic_regression')
            t_classifier_results.append(classifier_results)
            print "Finished %s." % label
        # Plot results
        plot_multi_top_weights('Tweets', t_classifier_results, num_top_features)

if __name__ == '__main__':
	main()
