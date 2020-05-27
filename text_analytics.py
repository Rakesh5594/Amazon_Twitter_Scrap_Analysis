#Run pip install nltk in cmd
#import nltk
#nltk.download()
import pandas as pd
import re               # Dealing with Regular Expressions.
import matplotlib.pyplot as plt     # plots.
#Download the wheel file uploaded in the LMS and place it in your sitepackages folder, Run the command <python -m pip install> from CMD by navigating to the sitepackages folder(C:\Users\your user name\AppData\Local\Programs\Python\Python36\Lib\site-packages)
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer     # Representation of Text as FEATURES (Document Term Matrix).
from sklearn.feature_extraction.text import TfidfVectorizer     # Representation of Text as FEATURES (TFIDF-Document Term Matrix).
from senticnet.senticnet import SenticNet
import nltk
#from amazon_dup import amazon

# Note: extract data from any data source (amazon as data source for this example) Before running the script.

#path = 'C:\\Users\\Jayendra Vadrevu\\Google Drive\\Darius\\1. DSA\\Course Material\\6. Text Analytics\\R and Python\R\\amazon_reviews.xlsx' # set reviews file path.
#raw_reviews = pd.read_excel(path,sheet_name="Reviews",names=['reviews']) # read reviews excel as pandas dataframe.
#url = "https://www.amazon.in/Avatar-Blu-ray-3D-Sam-Worthington/product-reviews/B01N9514ND/ref=cm_cr_arp_d_paging_btm_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"
url = "https://www.amazon.com/Interstellar-Matthew-McConaughey/product-reviews/B00TU9UO1W/ref=cm_cr_arp_d_paging_btm_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"
from amazon_scraper import amazon
reviews_list = amazon(url,10)
len(reviews_list)
raw_reviews = pd.DataFrame({'reviews': reviews_list})
raw_reviews.shape  # examine dimensions/shape of dataframe.
raw_reviews.head(10) # examine first n (i.e 10 in this case) rows of dataframe


from twitter_scraper import get_tweets
reviews_list_tw= (get_tweets("#interstellar", 'en', 'recent', 1000))
len(reviews_list_tw)
raw_reviews_tw = pd.DataFrame({'reviews': reviews_list_tw})
raw_reviews_tw.shape  # examine dimensions/shape of dataframe.
raw_reviews_tw.head(10) # examine first n (i.e 10 in this case) rows of dataframe
############### text cleaning function #############################

def text_clean_one(): #regular expressions
    for i in range(0, len(raw_reviews.reviews), 1):
        raw_reviews['reviews'].iloc[i] = re.sub("RT @[\w_]+: ", "", raw_reviews['reviews'].iloc[i])    #Removes RT @<username>:
        raw_reviews['reviews'].iloc[i] = re.sub("<.*?>", "", raw_reviews['reviews'].iloc[i])    # Removes HTML tags.
        raw_reviews['reviews'].iloc[i] = re.sub(r'[^\x00-\x7F]+', ' ', raw_reviews['reviews'].iloc[i]) #only ascii
        raw_reviews['reviews'].iloc[i] = re.sub(' +', ' ', raw_reviews['reviews'].iloc[i]) # replacing spaces to single space
        raw_reviews['reviews'].iloc[i] = raw_reviews['reviews'].iloc[i].lower() # converting to lower case
        raw_reviews['reviews'].iloc[i] = re.sub("[^\w\s]", "", raw_reviews['reviews'].iloc[i])  # Removes punctuations
        raw_reviews['reviews'].iloc[i] = re.sub('[^0-9a-zA-Z ]+', "", raw_reviews['reviews'].iloc[i])  # Keeps only alphanumeric
    return raw_reviews
################# end of function ##################################

raw_reviews.head(10)    # Before cleaning the data.

clean_reviews = text_clean_one()  # Cleaning Function.

clean_reviews.head(10)      # Examine data after cleaning.

len(clean_reviews)

# stopwords list preparation.
stopwords_user_file = open("D:\\Data Science\\DSA\\11.Text Analytics\\Python-NLTK\\stopwords.txt")  # locate user defined stopwords list file.
stopwords_user = set(stopwords_user_file.read().split())  # reading words from the file.
from nltk.corpus import stopwords
stopwords_english = set(stopwords.words('english'))     # inbuilt english stop words.
stopwords = list(stopwords_user.union(stopwords_english))   # Unique words from both the lists.
len(stopwords)

# removing stopwords from clean_reviews.
clean_reviews['without_stopwords'] = clean_reviews['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

clean_reviews_final = pd.DataFrame(clean_reviews.without_stopwords) # Dataframe with cleaned_reviews & removed stopwords.

clean_reviews_final.head(5)
len(clean_reviews_final)

#  Removal of Empty Reviews(Documents)
for j in range(1, len(clean_reviews_final), 1):
    if len(word_tokenize(str(clean_reviews_final.without_stopwords[j]))) < 1:
        clean_reviews_final = clean_reviews_final.drop([j])

len(clean_reviews_final)
# Transforming Reviews into DOCUMENT-TERM-MATRIX using CountVectorizer.

clean_reviews_series = clean_reviews_final.without_stopwords  #vectorizer needs a series object.

## N gram Analysis
#Note: CountVectorizer can handle Ngrams using 'ngram_range' arguement.
"""max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
max_df = 25 means "ignore terms that appear in more than 25 documents"""

"""min_df is used for removing terms that appear too infrequently. For example:
min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
min_df = 5 means "ignore terms that appear in less than 5 documents"."""

vectorizerng  = CountVectorizer(ngram_range=(1,2),min_df=0.01)      #  one and two grams( i.e unigrams and bigrams).

document_term_matrix_ng = vectorizerng.fit_transform(clean_reviews_series)      # DOCUMENT-TERM-MATRIX, page 15 in LMS

document_term_matrix_ng = pd.DataFrame(document_term_matrix_ng.toarray(), columns=vectorizerng.get_feature_names())    # DTM to Dataframe.

document_term_matrix_ng.shape

document_term_matrix_ng.head(10)

#word cloud frequencies of words
words = dict(document_term_matrix_ng.apply(sum , axis = 0))  ## this needs an dictionary object
wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(words)  #  fit_words() is used to plot wordcloud using dictionary.
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Create DTM-Document Term Matrix
vectorizer = CountVectorizer()     # Initiating CountVectorizer. ( with Default Parameters)

document_term_matrix = vectorizer.fit_transform(clean_reviews_series)      # DOCUMENT-TERM-MATRIX

document_term_matrix = pd.DataFrame(document_term_matrix.toarray(), columns=vectorizer.get_feature_names())    # DTM to Dataframe.

document_term_matrix.shape

document_term_matrix.head(10)


#  word cloud
#  wordcloud using frequencies of words
words = dict(document_term_matrix.apply(sum , axis = 0))  ## this needs a dictionary object
wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(words)  #  fit_words() is used to plot wordcloud using dictionary.
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Create DTM- with TF-IDF
vectorizeridf = TfidfVectorizer()     # Initiating CountVectorizer. ( with Default Parameters)

document_term_matrix_idf = vectorizeridf.fit_transform(clean_reviews_series)      # DOCUMENT-TERM-MATRIX

document_term_matrix_idf = pd.DataFrame(document_term_matrix_idf.toarray(), columns=vectorizeridf.get_feature_names())    # DTM to Dataframe.

document_term_matrix_idf.shape

document_term_matrix_idf.head(10)


#  word cloud
#  wordcloud using TFIDF of words, some of the docs some words keep repeating in all the docs, (Science) It works as log functions to normalize the words
words = dict(document_term_matrix_idf.apply(sum , axis = 0))  ## this needs an dictionary object
wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(words)  #  fit_words() is used to plot wordcloud using dictionary.
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#####  positive and negative words using user-built lexicons, plotting their wordclouds.
sn = SenticNet()
positive_words=[]
negative_words = []
for word in vectorizer.get_feature_names():
    if word in sn.data:
        if sn.polarity_value(word) == 'positive':
            positive_words.append(word)
        if sn.polarity_value(word) == 'negative':
            negative_words.append(word)

len(positive_words)
len(negative_words)


positive_words = dict(document_term_matrix[positive_words].apply(sum , axis = 0))
negative_words = dict(document_term_matrix[negative_words].apply(sum , axis = 0))

#  positive words wordcloud using frequency of words
wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(positive_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#  Negative words wordcloud using frequency of words
wordcloud = WordCloud(max_font_size=40,max_words = 50,background_color = "white").fit_words(negative_words) ##wordcloud using frequencies ( this needs an dictionary object)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Most positive and Most Negative Reviews


Review_polarity_df = pd.DataFrame()
for review in clean_reviews_final.without_stopwords :
    tokens = word_tokenize(review)
    count = 0
    pol = 0
    for word in tokens:
            if word in sn.data:
             pol = pol + float(sn.polarity_intense(word))
             count = count + 1
    if count > 0:
        polarity= pol / count
    else:
        polarity = 0
    temp_df= pd.DataFrame([review,polarity]).T
    Review_polarity_df = pd.concat([Review_polarity_df, temp_df],ignore_index=True)


Review_polarity_df.columns = ['review','polarity']

Reviews_Sorted= Review_polarity_df.sort_values( by =['polarity'], ascending = [False])

Most_positive_Reviews = Review_polarity_df.sort_values( by =['polarity'], ascending = [False])  # Most positive reviews.
Most_Negative_Reviews = Review_polarity_df.sort_values( by =['polarity'], ascending = [True])   # Most Negative reviews.


# End of Script.
Most_positive_Reviews
Most_Negative_Reviews

