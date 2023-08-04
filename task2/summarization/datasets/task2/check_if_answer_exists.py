import pdb
import json
import csv

input = "val_sum_t5-base_tuned.json"
output = "{}_check.csv".format(input.split('.json')[0])

with open(input, 'r') as inF, open(output, 'w') as outF:
    jf = json.load(inF)
    jf = jf["data"]
    csvwriter = csv.writer(outF)
    for line in jf:
        context = line["context"]
        answer = line["answers"]['text'][0]
        id = line["id"]
        answerInContext = context.find(answer)
        csvwriter.writerow([id, answerInContext, answer, context])

