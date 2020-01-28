# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:30:55 2020

@author: Noman
"""

#%%
import Basic_Models as bm
basic_model = bm.BasicModels()

import Threat_Embedding_Code as ta
dsthreat = ta.Threat_Dataset()


import Word_Embedding as we
w2v = we.WordEmbeddings()


#import bert_embedding as be


import pandas as pd
from gensim import models
from sklearn.model_selection import train_test_split

from gensim.models.wrappers import FastText
from gensim.models.fasttext import FastText, load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors


import numpy
import numpy as np
from numpy.random import seed
seed(1)




#%% load Vocabulary

wxTrain1, wxTest1, wyTrain1, wyTest1, sentences_corpus, keywords_dictionary, labels = dsthreat.Generate_Abusive_Ngrams(_ngram_range=(1,1), _max_features=None, words= True)

import Basic_Models as bm
basic_model = bm.BasicModels()
wcnn1d_ngram = basic_model.CNN1D_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1, _epochs = 50, _verbose=1)
wlstm_ngram = basic_model.bc_LSTM_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1,_epochs = 50, _verbose=1)
wbilstm_ngram = basic_model.Bidirectional_LSTM_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1, _epochs = 50, _verbose=2)


#%% BOW algo curves

import Basic_Models as bm
basic_model = bm.BasicModels()
ng_to_vec = 6
models = [wcnn1d_ngram,wlstm_ngram,wbilstm_ngram]
labels = ["BOW-1DCNN", "BOW-LSTM", "BOW-BiLSTM"]
xTest_data = [wxTest1,wxTest1,wxTest1]
yTest_data = [wyTest1,wyTest1,wyTest1]
basic_model.ROC_CURVE_ALL_MODEL_Diff_Data_2d_3d_Ngram(xTest_data, yTest_data,0, models,labels, _linestyle = ':', _figsize = (8,8))


#%%glove embeding testing
w2v_file_glove = "F:\\Machine_Learning\\Basic_Models\\glove_word2vec.txt"
wgTrain1, wgTest1, wgTrain1, wgTest1, sentences_corpus, keywords_dictionary, labels = dsthreat.Generate_Abusive_Ngrams(_ngram_range=(1,1), _max_features=100, words= True)

import Word_Embedding as we
w2v = we.WordEmbeddings()
gxTrain, gxTest, gyTrain, gyTest = w2v.generate_glovepretrained_w2v_doc_or_tweet_keywords(w2v_file_glove, keywords_dictionary, labels)#words are not necessary here
print(gxTrain.shape)


#%%glove embeding testing
import Basic_Models as bm
basic_model = bm.BasicModels()
gcnn1d_ngram = basic_model.CNN1D_Ngrams(gxTrain, gxTest, gyTrain, gyTest, _epochs = 50, _verbose=2)
glstm_ngram = basic_model.bc_LSTM_Ngrams(gxTrain, gxTest, gyTrain, gyTest,_epochs = 50, _verbose=5)
gbilstm_ngram = basic_model.Bidirectional_LSTM_Ngrams(gxTrain, gxTest, gyTrain, gyTest, _epochs = 50, _verbose=5)




#%% glove curve
import Basic_Models as bm
basic_model = bm.BasicModels()
ng_to_vec = 6
models = [gcnn1d_ngram,glstm_ngram,gbilstm_ngram]
labels = ["Glove-w2v-1DCNN", "Glove-w2v-LSTM", "Glove-w2v-BiLSTM"]
xTest_data = [gxTest,gxTest,gxTest]
yTest_data = [gyTest,gyTest,gyTest]
basic_model.ROC_CURVE_ALL_MODEL_Diff_Data_2d_3d_Ngram(xTest_data, yTest_data,0, models,labels, _linestyle = ':', _figsize = (8,8))







#%% fast load
wfTrain1, wfTest1, wfTrain1, wfTest1, sentences_corpus, keywords_dictionary, labels = dsthreat.Generate_Abusive_Ngrams(_ngram_range=(1,1), _max_features=100, words= True)

w2v_file_fast_text = "C:\FasText\\cc.ur.300.bin.gz"
w2vmodel = FastText.load_fasttext_format(w2v_file_fast_text)   
print("Word 2 Vector File Loaded!")      
vector = w2vmodel.wv['easy']
print( "Shape of Vector:" + str(vector.shape))
        




#%% fast algorithms 
X_train_Vector = []
for kl in keywords_dictionary:
    vector_list = []
    for word in kl:
        if word in w2vmodel.wv.vocab:
            vector_list.append(w2vmodel[word])
        else:
            vector_list.append(np.random.uniform(-0.1, 0.1, 300))
            
    matrix_2d = np.array(vector_list)
                #print(matrix_2d.shape)
    average_sentence_vector = np.mean(matrix_2d, axis = 0)
            
    X_train_Vector.append(average_sentence_vector)
        
X = numpy.array(X_train_Vector)
print( "Shape of training documents" + str(X.shape)) 

import Basic_Models as bm
basic_model = bm.BasicModels()  
fxTrain, fxTest, fyTrain, fyTest = train_test_split(X, labels, test_size = 0.2)

fcnn1d_ngram = basic_model.CNN1D_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_epochs = 50, _verbose=2)
flstm_ngram = basic_model.bc_LSTM_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_epochs = 50, _verbose=5)
fbilstm_ngram = basic_model.Bidirectional_LSTM_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_epochs = 50, _verbose=5)



#%% fasttext curve

import Basic_Models as bm
basic_model = bm.BasicModels()
ng_to_vec = 6
models = [fcnn1d_ngram,flstm_ngram,fbilstm_ngram]
labels = ["Fasttext-w2v-1DCNN", "Fasttext-w2v-LSTM", "Fasttext-w2v-BiLSTM"]
xTest_data = [fxTest,fxTest,fxTest]
yTest_data = [fyTest,fyTest,fyTest]
basic_model.ROC_CURVE_ALL_MODEL_Diff_Data_2d_3d_Ngram(xTest_data, yTest_data,0, models,labels, _linestyle = ':', _figsize = (8,8))




#%% mix curve

import Basic_Models as bm
basic_model = bm.BasicModels()
ng_to_vec = 6
models = [wbilstm_ngram,fcnn1d_ngram,gcnn1d_ngram]
labels = ["BOW-BiLSTM", "Fasttext-w2v-1DCNN","Glove-w2v-1DCNN"]
xTest_data = [wxTest1,fxTest,gxTest]
yTest_data = [wyTest1,fyTest,gyTest]
basic_model.ROC_CURVE_ALL_MODEL_Diff_Data_2d_3d_Ngram(xTest_data, yTest_data,0, models,labels, _linestyle = ':', _figsize = (8,8))






#%%bert embeding testing
#wxTrain1, wxTest1, wyTrain1, wyTest1, sentences_corpus, keywords_dictionary, labels = dsthreat.Generate_Abusive_Ngrams(_ngram_range=(1,1), _max_features=100, words= True)

#import bert_embedding as be
#be.Generate_Abusive_BERT_Embeddings(sentences_corpus, labels)

