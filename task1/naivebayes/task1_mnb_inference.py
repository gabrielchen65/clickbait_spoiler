import csv 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
import pickle
import json
from task1_naiveBayes import tokenize


def return_classifier(classifier):
    path = '/Users/krishthek/Documents/uWaterloo/msci641/project/naivebayes'

    with open(path+"/"+classifier+'.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(path+"/"+classifier+'_vec.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open(path+'/label_enc.pkl', 'rb') as f:
        label_enc = pickle.load(f)
    
    return clf, vectorizer, label_enc

def run_inference(input_file, output_file):
    clf, vectorizer, label_enc = return_classifier('mnb_bi')
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        out.write("id,spoilerType\n")
        test_arr = []
        for i in inp:
            i = json.loads(i)
            # print(str(i["id"]))
            token = tokenize(str(i['targetTitle']))
            combinedRow = [' '.join(token)]
            # print(combinedRow)
            # print(str(i['id']))
            X_vectorized = vectorizer.transform(combinedRow)
            y_pred = clf.predict(X_vectorized)
            y_pred_real = label_enc.inverse_transform(y_pred)
            # print(combinedRow, y_pred_real)
            out.write(str(i['id'])+ ',' + y_pred_real[0] + '\n')
            

def main():
    test_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/clickbait-detection-msci641-s23/test.jsonl'
    output_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/naivebayes/mnb_bi_out.csv'
    run_inference(test_file, output_file)
    pass

if __name__ == '__main__':
    main()