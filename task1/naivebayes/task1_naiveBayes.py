import csv
import re
import random
import sys
import os
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pickle
import sklearn.metrics


def return_data(file):
    postText_arr = []
    tags_arr=[]
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        # print(result['targetTitle'])
        # postText_arr.extend(result['postText'])
        postText_arr.append(result['targetTitle'])
        tags_arr.extend(result['tags'])
    # print(postText_arr)
    return postText_arr, tags_arr


def tokenize(line):
    ''' Input is a line.
    Tokenizes and returns line as array where each element is a token.
    There is no stop word removal in this function.
    '''
    # tokenLine = [] #to store every token
    # print(line)
    line=line.lower()
    splitLine = re.split('\W+', line)
    tokenizedLine =[]
    for i in range(len(splitLine)):
        if(splitLine[i]!=''):
            tokenizedLine.append(splitLine[i])
    return tokenizedLine

def main():
    train_file = '../task1/clickbait-detection-msci641-s23/train.jsonl'
    val_file = '../task1/clickbait-detection-msci641-s23/val.jsonl'
    train_text, train_label = return_data(train_file)
    val_text, val_label = return_data(val_file)
    print(len(val_text))
    X_train = []
    X_val = []
    for line in train_text:
        tokens = tokenize(line)
        combinedRow = ' '.join(tokens)
        # print(combinedRow)
        X_train.append(combinedRow)
        # X_train.append(tokens)
    # print(X_train)
    for line in val_text:
        tokens = tokenize(line)
        combinedRow = ' '.join(tokens)
        # print(combinedRow)
        X_val.append(combinedRow)
        # X_train.append(tokens)
    # print(len(X_val))

    save_path = ''

    unigram_vectorizer = CountVectorizer(ngram_range=(1,1))
    bigram_vectorizer = CountVectorizer(ngram_range=(2,2))
    mixed_vectorizer = CountVectorizer(ngram_range=(1,2))

    X_train_unigram = unigram_vectorizer.fit_transform(X_train)
    X_train_bigram = bigram_vectorizer.fit_transform(X_train)
    X_train_mixed = mixed_vectorizer.fit_transform(X_train)

    X_val_unigram = unigram_vectorizer.transform(X_val)
    X_val_bigram = bigram_vectorizer.transform(X_val)
    X_val_mixed = mixed_vectorizer.transform(X_val)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_label)
    y_val_enc = label_encoder.transform(val_label)
    # y_test_enc = label_encoder.transform(y_test)
    # print(np.count_nonzero(y_val_enc== 2))
    one =0
    two =0
    zero = 0
    for i in y_val_enc:
        if i==0:
            zero=zero+1
        if i==1:
            one = one+1
        if i==2:
            two = two+1
    # print(zero,one,two)

    clf_uni = MultinomialNB()
    print("unigram")
    clf_uni.fit(X_train_unigram, y_train_enc)
    print(clf_uni.score(X_val_unigram, y_val_enc))
    y_pred = clf_uni.predict(X_val_unigram)
    # print(sklearn.metrics.f1_score(X_val_unigram, y_val_enc, average="macro"), labels=[0,1,2])
    print(sklearn.metrics.precision_recall_fscore_support(y_val_enc,y_pred, average = 'weighted'))
    
    print("bigram")
    clf_bi = MultinomialNB()
    clf_bi.fit(X_train_bigram, y_train_enc)
    print(clf_bi.score(X_val_bigram, y_val_enc))
    y_pred = clf_bi.predict(X_val_bigram)
    print(sklearn.metrics.precision_recall_fscore_support(y_val_enc,y_pred))

    clf_mixed = MultinomialNB()
    clf_mixed.fit(X_train_mixed, y_train_enc)
    print("Mixed")
    print(clf_mixed.score(X_val_mixed, y_val_enc))
    y_pred = clf_mixed.predict(X_val_mixed)
    print(sklearn.metrics.precision_recall_fscore_support(y_val_enc,y_pred))
    
    # with open(save_path+'mnb_bi.pkl', 'wb') as f: 
    #     pickle.dump(clf_bi, f)
    # with open(save_path+'mnb_bi_vec.pkl', 'wb') as f: 
    #     pickle.dump(bigram_vectorizer, f)
    # with open(save_path+'label_enc.pkl', 'wb') as f: 
    #     pickle.dump(label_encoder, f)



if __name__ == '__main__':
    main()