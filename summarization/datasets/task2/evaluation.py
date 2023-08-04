import pdb
import json
import evaluate

input = "val_4summary_multi-concat.json"
answerFile = 'summarization-output-multi.txt'
output = 'sum-pred-ref.txt'
predictions = []
references = []

with open(input, 'r') as inF, open(output, 'w') as outF, open(answerFile, 'r') as aF:
    jf = json.load(inF)
    afLines = aF.readlines()
    for idx, line in enumerate(jf['data']):
        references += jf['data'][idx]["answers"]['text']
        predictions += [afLines[idx][:-2]]
 
metric = evaluate.load("meteor")

print(metric.compute(predictions=predictions, references=references))

