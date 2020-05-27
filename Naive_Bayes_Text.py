
# model building with text analytics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



# read mails as dataframe.
path = 'C:\\Users\\Jayendra Vadrevu\\PycharmProjects\\MyPy\\Text_Analytics\\SMS Data\\sms.tsv'
sms = pd.read_table(path, header=None, names=['label', 'message'])


sms.shape  # examine the shape


sms.head(10)  # examine the first 10 rows


sms.label.value_counts()  # examine the class distribution

sms['label_num'] = sms.label.map({'ham':0, 'spam':1})  # convert label to a numerical variable.

sms.head(10)  # check data after conversion.

# defining X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)
len(X)


#  splitting data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# instantiate the vectorizer
vect = CountVectorizer()


################## equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
################################

# examine the document-term matrix
X_train_dtm


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# Building and evaluating a model
#
# We will use [multinomial Naive Bayes]
# import and instantiate a Multinomial Naive Bayes model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

# End of script.

