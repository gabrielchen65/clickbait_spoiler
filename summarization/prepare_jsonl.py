import json
import pdb

input = 'train.jsonl'
extension = '.jsonl'
out_phrase = input.replace(extension, "_phrase"+extension)
out_passage = input.replace(extension, "_passage"+extension)
out_multi = input.replace(extension, "_multi"+extension)

keepAttribute = ['uuid', 'targetParagraphs', 'targetTitle', 'spoiler','tags']
with open(input, 'r') as inF, open(out_phrase, 'w') as outFPh, open(out_passage, 'w') as outFPa, open(out_multi, 'w') as outFMul:
    jsonList = list(inF)
    for line in jsonList:
        jsonLine = json.loads(line)
        data = {}
        for key, value in jsonLine.items():
            if key in keepAttribute:
                data[key] = value
        if jsonLine['tags'] == ["phrase"]:
            json.dump(data, outFPh)
            outFPh.write('\n')
        elif jsonLine['tags'] == ["passage"]:
            json.dump(data, outFPa)
            outFPa.write('\n')
        elif jsonLine['tags'] == ["multi"]:
            json.dump(data, outFMul)
            outFMul.write('\n')