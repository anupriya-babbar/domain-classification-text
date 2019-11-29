# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# path_train : location of test file
# Code starts here
df=pd.read_csv(path_train)
df.head()
category=[]
def label_race(row):
#for row in range(len(df)):
    if row['food'] == "T":
        return 'food'
       #category.append('food')
    elif row['recharge'] == "T":
       return 'recharge'
       #category.append("recharge")
    elif row['support']=='T':
        return 'support'
        #category.append("support")
    elif row['reminders']=="T":
        #category.append('reminders')
        return 'reminders'
    elif row['nearby']=='T':
        return 'nearby'
        #category.append('nearby')
    elif row['movies']=='T':
        return 'movies'
        #category.append('movies')
    elif row['casual']=='T':
        return 'casual'
        #category.append('casual')
    elif row['other']=='T':
        return 'other'
        #category.append('other')
    elif row['travel']=='T':
        return 'travel'
        #category.append('travel')
#Index_label = df[df.iloc[::]=='T'].index.tolist() 
df['category'] = df.apply(label_race, axis=1)
df=df.drop(['food','recharge','support','reminders','nearby','movies','casual','other','travel'],1)
    #b = (df.ix[row.name] == row['value'])
    #return b
#category



# --------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
# Sampling only 1000 samples of each category
#msg=df['message']
df = df.groupby('category').apply(lambda x: x.sample(n=1000, random_state=0))
#df
# Code starts here
#all_text=df['message']
#all_text=all_text.lower()
all_text=df['message'].str.lower()
tfidf=TfidfVectorizer(stop_words='english')
X=tfidf.fit_transform(all_text).toarray()
#X=tfidf.transform(df['message'])
#X=X.toarray()
le=preprocessing.LabelEncoder()
y=le.fit_transform(df['category'])
#print(all_text)


# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.3,random_state=42)
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_val)
log_accuracy=accuracy_score(y_val,y_pred)
print(log_accuracy)
nb=MultinomialNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_val)
nb_accuracy=accuracy_score(y_val,y_pred)
lsvm=LinearSVC(random_state=0)
lsvm.fit(X_train,y_train)
y_pred=lsvm.predict(X_val)
lsvm_accuracy=accuracy_score(y_val,y_pred)
print(lsvm_accuracy)


# --------------
# path_test : Location of test data

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = df_test.apply (lambda row: label_race (row),axis=1)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Code starts here
all_text=df_test['message'].str.lower()
#tfidf=TfidfVectorizer(stop_words='english')
X_test=tfidf.transform(all_text).toarray()
#X=tfidf.transform(df['message'])
#X=X.toarray()
#le=preprocessing.LabelEncoder()
y_test=le.transform(df_test['category'])
y_pred=log_reg.predict(X_test)
log_accuracy_2=accuracy_score(y_test,y_pred)
y_pred=nb.predict(X_test)
nb_accuracy_2=accuracy_score(y_test,y_pred)
y_pred=lsvm.predict(X_test)
lsvm_accuracy_2=accuracy_score(y_test,y_pred)
#print(all_text)


# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]

# Code starts here
dictionary=corpora.Dictionary(doc_clean)
doc_term_matrix=[dictionary.doc2bow(doc) for doc in doc_clean]
lsimodel=LsiModel(corpus=doc_term_matrix, num_topics=5, id2word=dictionary)
pprint(lsimodel.print_topics(5))


# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return topic_list, coherence_values


# Code starts here
topic_list,coherence_value_list=compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=doc_clean, start=1, limit=41, step=5)
max_index=coherence_value_list.index(max(coherence_value_list))
opt_topic= topic_list[max_index]
lda_model=LdaModel(corpus=doc_term_matrix, num_topics=opt_topic, id2word = dictionary,iterations=10 , passes=30 ,random_state=0)
pprint(lda_model.print_topics(5))


