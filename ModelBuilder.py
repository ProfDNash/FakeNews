"""
MODEL BUILDER
Created on Wed May 20 11:08:45 2020
Goal: Use processed data to build predictive models      

@author: David A. Nash
"""
import numpy as np ##linear algebra tools
import pandas as pd ##data processing, CSV I/O (e.g. pd.read_csv)
from sklearn import cluster ##for k-means clustering
from sklearn.ensemble import RandomForestClassifier #as RF ##random forest classifier
from sklearn.model_selection import train_test_split ##split data into train and test sets
##Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA ##PCA for visualizing clustering
from sklearn.manifold import TSNE ##TSNE for visualizing clustering

articleList = np.load('Articles.npy').tolist()
X = np.load('X.npy')  ##np.array whose rows are average Word2Vec vectors for each article
y = np.load('y.npy')  ##np.array of labels 0=fake, 1=True

##Use k-means clustering (unsupervised learning) to attempt to split the data
kmeans = cluster.KMeans(n_clusters=2, verbose=True) ##one cluster each for fake and true
##note that the labels may be backwards as we have no control over which cluster is which
clustered = kmeans.fit_predict(X) ##fit_predict will return predicted labels
##create dataframe with X, y, and the predicted labels
testDF = {'Article Vector': articleList, 'Labels': y, 'Prediction': clustered}
testDF = pd.DataFrame(data=testDF)

correct = 0
incorrect = 0
for index, row in testDF.iterrows():
    if row['Labels'] == row['Prediction']:
        correct+=1
    else:
        incorrect+=1

if correct/(correct+incorrect)<0.5: ##then the labels are backwards in the clustering
    print('Cluster 0 most closely matches True, while Cluster 1 most closely matches Fake')
    print("Correctly clustered news: " + str((incorrect*100)/(correct+incorrect)) + '%')
else:
    print('Cluster 0 most closely matches Fake, while Cluster 1 most closely matches True')
    print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + '%')


input("Press enter to train a Random Forest Classifier")
##Try RF classifier (supervised learning) next
##Split into 90/10 train/test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.1)

def findRFparams(Xtrain,Xtest,ytrain,ytest):
    bestN = 10
    bestDepth = 6
    bestAcc = 0
    ##find a good choice of parameters for RF (already tested d=6,7,8,9)
    for n in [10, 20, 50, 100, 200]:
        print ('n =',n)
        for d in [10, 11, 12]:
            RF = RandomForestClassifier(n_estimators=n, max_depth=d)
            RF.fit(Xtrain,ytrain)
            Acc = RF.score(Xtest,ytest)
            if Acc > bestAcc:
                bestN=n
                bestDepth=d
                bestAcc=Acc
    return bestN, bestDepth, bestAcc




