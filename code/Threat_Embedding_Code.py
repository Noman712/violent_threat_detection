# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:20:44 2019

@author: Noman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:45:52 2019

@author: Noman
"""


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
# Import adjustText, initialize list of texts
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import itertools
import re
import os
from IPython.display import SVG
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import preprocessor as p

import nltk
from nltk.corpus import stopwords
stopword = stopwords.words('english')


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import os
import csv


#Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

import natsort 
from cleantext import clean


class Threat_Dataset:

    def avg_word(sentence):
        words = sentence.split()
        return (sum(len(word) for word in words)/len(words))
    def Clean_Text(text):
        data = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=True,      # replace all currency symbols with a special token
        no_punct=True,                 # fully remove punctuation
        replace_with_url=" ",
        replace_with_email=" ",
        replace_with_phone_number=" ",
        replace_with_number=" ",
        replace_with_digit=" ",
        replace_with_currency_symbol=" ",
        lang="en"                       # set to 'de' for German special handling
        )
        return data
    
    def Generate_Abusive_Ngrams(self, _ngram_range=(1,1), _max_features=5000, words = True ):    
        
        
        dataset_path = r"F:\threat\violent_threats_dataset.csv"
        #dataset_path = r"F:\RazaResults\suicide\suicide_dataset_2020.csv"
    
        #load data
        df = pd.read_csv(dataset_path, usecols=['Tweets','Labels'],sep=',')
        df = df.drop_duplicates(keep=False)
        #print (df.head())
        #print(df.shape)
        
        
        df['Tweets']=df['Tweets'].apply(Threat_Dataset.Clean_Text)
        
        print (df.head(10))
        
        yeswisedf = df[(df['Labels'] == 1)]
        print(yeswisedf.head())
        
        yeswisedf['word_count'] = yeswisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        yeswisedf[['Tweets','word_count']].head()
        
        yeswisedf['char_count'] = yeswisedf['Tweets'].str.len() ## this also includes spaces
        yeswisedf[['Tweets','char_count']].head()
        
        
        yeswisedf['avg_word'] = yeswisedf['Tweets'].apply(lambda x:  Threat_Dataset.avg_word(x))
        yeswisedf[['Tweets','avg_word']].head()
        
        
                
        print(yeswisedf.describe())
        print(yeswisedf.sum(axis=0) )
        #yeswisedf.to_csv(r"F:\threat\threats_only__dataset.csv", sep=',', encoding='utf-8', index=False)
        #print("File Saved!")  


        nowisedf = df.loc[df['Labels'] == 0]
        print(nowisedf.head())
        
        nowisedf['word_count'] = nowisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        nowisedf[['Tweets','word_count']].head()
        
        nowisedf['char_count'] = nowisedf['Tweets'].str.len() ## this also includes spaces
        nowisedf[['Tweets','char_count']].head()
        
        
        nowisedf['avg_word'] = nowisedf['Tweets'].apply(lambda x:  Threat_Dataset.avg_word(x))
        nowisedf[['Tweets','avg_word']].head()
        
        print(nowisedf.describe())
        print(nowisedf.sum(axis=0) )
        
        #nowisedf = nowisedf[:len(yeswisedf)]
        
        
        df_row = yeswisedf.append(nowisedf, ignore_index=True)#pd.concat([yeswisedf['Tweets'], nowisedf['Tweets']], axis=0)
        #print(type(df_row))
        #print(df_row.head())
        #print(df_row.shape)
        
        
        print("Total Dataset:" + str(len(df_row)))
        print("yes class:" + str(len(yeswisedf)))
        print("no class:" + str(len(nowisedf)))
        
        
        rows = len(df_row)
        
        Number_OF_Documents = rows + 1
        #Number_OF_Documents = len(yeswisedf) + len(nowisedf)
        Number_OF_POSITIVE_SAMPLES = len(yeswisedf)
        Number_OF_NEGATIVE_SAMPLES = len(nowisedf)
        
        
        
        y_train = np.empty(Number_OF_Documents-1)
       
        for i in range(0,Number_OF_Documents-1):
            if i < Number_OF_POSITIVE_SAMPLES - 1:
                y_train[i] = 1
            else:
                y_train[i] = 0
       
        
        
        keywords_dictionary = []
        sentences_corpus = []
        #for index,row in df.iterrows(): 
        for index,row in df_row.iterrows():
                text = str(row['Tweets'])
                
                sentences_corpus.append(text)
                list_of_words = text.split(" ")
                keywords_dictionary.append(list_of_words)
            
        
        
        print(keywords_dictionary[0])
        print(sentences_corpus[0])
        
        
        vocab = []
        for kl in keywords_dictionary: 
            for w in kl:
                vocab.append(str(w))
            
        print(len(vocab))
        
        
        corpus = []
        for kl in sentences_corpus: 
            corpus.append(''.join(kl))
        
        print(len(corpus))
        print(corpus[0])
        
        
    
        if words:
            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features)
            Count_Vect = vectorizer.fit_transform(vocab)
            #print(vectorizer.get_feature_names())
            #print(X.toarray())
        else:
            vectorizer = CountVectorizer(ngram_range=_ngram_range, token_pattern = r"(?u)\b\w+\b",  analyzer='char')
            Count_Vect = vectorizer.fit_transform(vocab)
            #print(vectorizer.get_feature_names())

                
                
        vectorizer = TfidfVectorizer(ngram_range=_ngram_range,max_features=_max_features) # You can still specify n-grams here.
        X = vectorizer.fit_transform(corpus).toarray()   
        #X = vectorizer.fit_transform(corpus)    
            
            
        print( "Shape of final Ngram vector:" + str(X.shape))
        print( "Shape of labels:" + str(y_train.shape))
        print(y_train[0:10])
        xTrain, xTest, yTrain, yTest = train_test_split(X, y_train, test_size = 0.3)   
        return xTrain, xTest, yTrain, yTest, sentences_corpus, keywords_dictionary, y_train
    


     

