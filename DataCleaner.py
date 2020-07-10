"""
DATA CLEANER
Created on Wed May 20 10:08:45 2020
Goal: Clean and preprocess data from fake and true news stories to create a dataset
      Each article will be encoded by the average of word vectors from Word2Vec      

@author: David A. Nash
"""
import numpy as np ##linear algebra tools
import pandas as pd ##data processing, CSV I/O (e.g. pd.read_csv)
import string # python library
import re # regex library
## Preprocesssing functions from gensim
## preprocess_string, strip_tags, strip_punctuation, strip_numeric,
## strip_multiple_whitespaces, remove_stopwords, strip_short
import gensim.parsing.preprocessing as pp
from gensim.models import Word2Vec # Word2vec


fakeData = pd.read_csv(r'Fake.csv')
trueData = pd.read_csv(r'True.csv')
##columns are 'title', 'text', 'subject', 'date'
##true stories have '(Reuters) - ' tags and or disclaimers (see below)
## those need to be removed
'''
The following statements were posted to the verified Twitter accounts
of U.S. President Donald Trump, @realDonaldTrump and @POTUS.
The opinions expressed are his own. Reuters has not edited 
the statements or confirmed their accuracy.  @realDonaldTrump : -
'''
##Note, none of the fake news stories contain '@realDonaldTrump : - ' or '(Reuters) - '
cleanedData = []
for data in trueData.text:
    if '@realDonaldTrump : - ' in data:
        data = data.split('@realDonaldTrump : - ')[1]
    if '(Reuters) - ' in data:
        data = data.split('(Reuters) - ')[1]
    cleanedData.append(data)
trueData.text = cleanedData

##combine the titles and text into a single string
trueData['sentences'] = trueData.title + ' ' + trueData.text
fakeData['sentences'] = fakeData.title + ' ' + fakeData.text

##Add labels for the data so that they can be combined into a single dataset
trueData['label'] = 1
fakeData['label'] = 0
finalData = pd.concat([trueData,fakeData])

# Randomize the rows so its all mixed up
finalData = finalData.sample(frac=1).reset_index(drop=True)

# Drop 'title', 'text', 'subject', and 'date' which we no longer need
finalData = finalData.drop(['title', 'text', 'subject', 'date'], axis = 1)
##save final sentence strings for later use
np.save('Articles', finalData.sentences.tolist())
##preprocess each sentence
def ProcessArticles(dat):
    processedData = []
    for sentence in dat:
        processedData.append(pp.preprocess_string(sentence))
    return processedData

processedData = ProcessArticles(finalData.sentences)
processedLabels = finalData.label
    
# Word2Vec model trained on processed data
model = Word2Vec(processedData, min_count=1)

# Getting the vector of an article based on average of all the word vectors in the article
# We get the average as this accounts for different article lengths

def ReturnVector(x):
    try:
        return model.wv.__getitem__(x)
    except:
        return np.zeros(100)  ##Word2Vec returns vectors of length 100 by default
    
def Article_Vector(article):
    word_vectors = list(map(lambda x: ReturnVector(x), article))
    return np.average(word_vectors, axis=0).tolist()

X = []
for article in processedData:
    X.append(Article_Vector(article))
##convert to np.arrays for use in various algorithms
X = np.array(X)
y = np.array(processedLabels)
np.save('X',X)
np.save('y',y)


