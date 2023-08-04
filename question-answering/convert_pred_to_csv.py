import pdb
import csv
import json

input_file = './datasets/predict_predictions.json'
output_file = './datasets/upload.csv'

with open(input_file, 'r') as inF, open(output_file, 'w') as outF:
    jsonData = json.load(inF)
    csvwriter = csv.writer(outF)
    csvwriter.writerow(['id', 'spoiler'])
    for k, v in jsonData.items():
        csvwriter.writerow([k,v])  
