# FakeNews
The goal of this project is to build a predictive model for identifying Fake News online.  
I will be building my initial model with the Fake News Dataset from Kaggle: 
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
(That dataset consists of two files True.csv and Fake.csv)

My initial testing with a Random Forest classifier trained on a 90/10 train/test split
of the data was able to correctly classify over 97% of the test data, however it does
not seem to scale well to new articles found "in the wild."  As such, I plan to enlarge
the training set that I have to work with.  One additional source is also from Kaggle:
https://www.kaggle.com/c/fake-news/overview
(Pulled the train.csv file and renamed it FNdata2.csv)

