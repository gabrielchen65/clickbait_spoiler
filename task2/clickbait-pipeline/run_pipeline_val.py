import pdb
import json
import csv
import subprocess
import shlex
import evaluate


def main():
  #----- split dataset according to classification ------#
  multiNonMulti = './datasets/val_multi_nonmulti.csv'
  valPath = './datasets/val_4summary_all-concat.json'
  nonMulti = './datasets/val_cla-pred_phrase-passage.json'
  multi = './datasets/val_cla-pred_multi-concat.json'
  with open(multiNonMulti, 'r') as f, open(valPath, 'r') as valF, \
       open(nonMulti, 'w') as nonF, open(multi, 'w') as mulF:
      csvreader = csv.reader(f)
      valF = json.load(valF)['data']
      nonMultiJson = {"data":[]}
      multiJson = {"data":[]}
      headerRow = True
      for row in csvreader:
        if headerRow: 
          headerRow = False
          continue
        id = int(row[0])
        if row[1] == 'multi':
           multiJson['data'].append(valF[id])
        elif row[1] == 'non-multi':
           nonMultiJson['data'].append(valF[id])
      json.dump(multiJson, mulF)
      json.dump(nonMultiJson, nonF)

        
  #----- run non-multi evaluation with the qa model ------#
  qa_script = "python3 ../question-answering/run_qa.py \
              --model_name_or_path ../question-answering/tmp/deepset-roberta-base-squad2-768/checkpoint-12000/ \
              --validation_file ./datasets/val_cla-pred_phrase-passage.json \
              --do_eval \
              --evaluation_metric meteor \
              --per_device_train_batch_size 12 \
              --max_seq_length 768 \
              --doc_stride 128 \
              --output_dir ./pipline_output/qa/ "
  #subprocess.call(shlex.split(qa_script))

  #----- prepare QA result ------#
  inputPre = "./pipline_output/qa/eval_predictions.json"
  inputRef = "./datasets/val_cla-pred_phrase-passage.json"
  outputF = "./pipline_output/qa/qa-pred-ref.json"

  with open(inputPre, 'r') as inPreF, open(inputRef, 'r') as inRefF, open(outputF, 'w') as outF:
      jPre = json.load(inPreF)
      jRef = json.load(inRefF)["data"]
      outJson = {}

      for element in jRef:
         id = element['id']
         outJson[id] = {
            "predictions": jPre[str(id)],
            "references": element['answers']['text'][0]
         }
      json.dump(outJson, outF, indent=4)


  #----- run multi evaluation with the summarization model ------#
  sum_script = "python3 ../summarization/run_summarization.py \
                --model_name_or_path ../summarization/tmp/Einmalumdiewelt-T5-Base_GNAD-30epoch \
                --do_predict \
                --test_file ./datasets/val_cla-pred_multi-concat.json \
                --text_column article \
                --summary_column highlights \
                --dataset_config '3.0.0' \
                --output_dir ./pipline_output/sum/ \
                --per_device_train_batch_size=4 \
                --per_device_eval_batch_size=4 \
                --overwrite_output_dir \
                --predict_with_generate \
                --evaluation_metric rouge \
                --max_target_length 70 \
                --source_prefix 'summarize: '"
  #subprocess.call(shlex.split(sum_script))

  #----- prepare summarization result ------#
  input = "./datasets/val_cla-pred_multi-concat.json"
  answerFile = './pipline_output/sum/generated_predictions.txt'
  output = './pipline_output/sum/sum-pred-ref.json'
  predictions = []
  references = []

  with open(input, 'r') as inF, open(output, 'w') as outF, open(answerFile, 'r') as aF:
      jf = json.load(inF)
      afLines = aF.readlines()
      outJson = {}
      for idx, line in enumerate(jf['data']):
          references += jf['data'][idx]["answers"]['text']
          predictions += [afLines[idx][:-2]]
          origianlID = jf['data'][idx]['id']
          outJson[origianlID] = {
             "predictions": afLines[idx][:-2],
             "references": jf['data'][idx]["answers"]['text'][0]
          }
      json.dump(outJson, outF, indent=4)
  
  """   metric = evaluate.load("meteor")
  print("Meteor score for Multi only:")
  print(metric.compute(predictions=predictions, references=references)) """

  #----- evaluate all ------#
  predictions = []
  references = []
  inputQA = "./pipline_output/qa/qa-pred-ref.json"
  inputSum = "./pipline_output/sum/sum-pred-ref.json"
  output = './pipline_output/all-pred-ref.csv'

  with open(inputQA, 'r') as in1F, open(inputSum, 'r') as in2F, open(output, 'w') as outF:
    qaJson = json.load(in1F)
    sumJson = json.load(in2F)
    for i in range(400):
      keyStr = str(i)
      if keyStr in qaJson.keys():
        predictions += [qaJson[keyStr]['predictions']]
        references += [qaJson[keyStr]['references']]
      elif keyStr in sumJson.keys():
        predictions += [sumJson[keyStr]['predictions']]
        references += [sumJson[keyStr]['references']]         

    metric = evaluate.load("meteor")
    print("Meteor score for all:")
    print(metric.compute(predictions=predictions, references=references))

if __name__ == "__main__":
    main()