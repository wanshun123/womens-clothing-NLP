#import textblob
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.classifiers import NaiveBayesClassifier
import nltk
#opinion = TextBlob("movie was good.")
#opinion.words
#opinion.sentiment

import numpy as np
import pandas as pd
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# df['Title'].fillna(' ')

# check values of 'Recommended IND'
np.unique(df["Recommended IND"].values)

# for this example just get rid of all NA
df = df.dropna()

# for textblob - necessary?
df['Recommended IND'] = np.where(df['Recommended IND'] == 1, 'pos', 'neg')

# combine title and review text
df['test'] = df['Title'] + ' ' + df['Review Text']

# split into 80% train 20% test
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

subset_train = train[['test', 'Recommended IND']]
tuples_train = [tuple(x) for x in subset_train.values]
subset_test = test[['test', 'Recommended IND']]
tuples_test = [tuple(x) for x in subset_test.values]

cl = NaiveBayesClassifier(tuples_train)

# [x[0] for x in tuples_test]
# cl.classify(tuples_test[10][0])

# cl.classify("This is an amazing library!")

predicted_classifications = []

def make_predictions():
    for i in range(len(tuples_test)):
        classification = cl.classify(tuples_test[i][0])
        predicted_classifications.append(classification)
    
make_predictions()
    
# predicted_classifications.count('pos')

# predicted_classifications[0] == tuples_test[0][1]

def accuracy():
    countCorrect = 0
    for i in range(len(tuples_test)):
        if predicted_classifications[i] == tuples_test[i][1]:
            countCorrect += 1
    accuracy = (countCorrect / len(predicted_classifications)) * 100
    return accuracy
    
accuracy()
    
def return_incorrect():
    incorrectIndexes = []
    for i in range(len(tuples_test)):
        if predicted_classifications[i] != tuples_test[i][1]:
            incorrectIndexes.append(i)
    return incorrectIndexes
    
return_incorrect()

# tuples_test[97]