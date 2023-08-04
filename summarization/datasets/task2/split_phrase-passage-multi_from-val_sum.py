import pdb
import json

input = 'val_sum_t5-base_tuned.json'
ref = 'val.jsonl'
outPh = input.replace('.json', '_phrase.json')
outPa = input.replace('.json', '_passage.json')
outMl = input.replace('.json', '_multi.json')

with open(input, 'r') as inF, open(outPh, 'w') as outFPh, \
     open(outPa, 'w') as outFPa, open(outMl, 'w') as outFMl, \
     open(ref, 'r') as reF:
  inJson = json.load(inF)
  reFList = list(reF)
  dataPh = {"data":[]}
  dataPa = {"data":[]}
  dataMl = {"data":[]}

  for idx, element in enumerate(inJson["data"]):
    refLine = json.loads(reFList[idx])
    tag = refLine['tags']
    if tag == ['phrase']:
      dataPh["data"].append(element)
    elif tag == ['passage']:
      dataPa["data"].append(element)
    elif tag == ['multi']:
      dataMl["data"].append(element)
    else:
      raise Exception("What")
  
  json.dump(dataPh, outFPh)
  json.dump(dataPa, outFPa)
  json.dump(dataMl, outFMl)
