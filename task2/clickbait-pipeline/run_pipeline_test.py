import pdb
import json
import csv
import subprocess
import shlex
import evaluate


def main():
  #----- split dataset according to classification ------#
  multiNonMulti = './datasets/test_multi_nonmulti.csv'
  testPath = './datasets/test_4summary_all-concat.json'
  nonMulti = './datasets/test_cla-pred_phrase-passage.json'
  multi = './datasets/test_cla-pred_multi-concat.json'
  with open(multiNonMulti, 'r') as f, open(testPath, 'r') as testF, \
       open(nonMulti, 'w') as nonF, open(multi, 'w') as mulF:
      csvreader = csv.reader(f)
      testF = json.load(testF)['data']
      nonMultiJson = {"data":[]}
      multiJson = {"data":[]}
      headerRow = True
      for row in csvreader:
        if headerRow: 
          headerRow = False
          continue
        id = int(row[0])
        if row[1] == 'multi':
           multiJson['data'].append(testF[id])
        elif row[1] == 'non-multi':
           nonMultiJson['data'].append(testF[id])
      json.dump(multiJson, mulF)
      json.dump(nonMultiJson, nonF)

        
  #----- run non-multi evaluation with the qa model ------#
  qa_script = "python3 ../question-answering/run_qa.py \
              --model_name_or_path ../question-answering/tmp/deepset-roberta-base-squad2-768/checkpoint-12000/ \
              --test_file ./datasets/test_cla-pred_phrase-passage.json \
              --do_predict \
              --evaluation_metric meteor \
              --per_device_train_batch_size 12 \
              --max_seq_length 768 \
              --doc_stride 128 \
              --output_dir ./pipline_output/qa/test/ "
  #subprocess.call(shlex.split(qa_script))


  #----- run multi evaluation with the summarization model ------#
  sum_script = "python3 ../summarization/run_summarization.py \
                --model_name_or_path ../summarization/tmp/Einmalumdiewelt-T5-Base_GNAD-30epoch \
                --do_predict \
                --test_file ./datasets/test_cla-pred_multi-concat.json \
                --text_column article \
                --summary_column highlights \
                --dataset_config '3.0.0' \
                --output_dir ./pipline_output/sum/test/ \
                --per_device_train_batch_size=4 \
                --per_device_eval_batch_size=4 \
                --overwrite_output_dir \
                --predict_with_generate \
                --evaluation_metric rouge \
                --max_target_length 70 \
                --source_prefix 'summarize: '"
  #subprocess.call(shlex.split(sum_script))
  #----- prepare summarization result ------#
  input = "./datasets/test_cla-pred_multi-concat.json"
  answerFile = './pipline_output/sum/test/generated_predictions.txt'
  output = './pipline_output/sum/test_sum-pred.json'

  with open(input, 'r') as inF, open(answerFile, 'r') as aF, open(output, 'w') as outF:
      jf = json.load(inF)
      afLines = aF.readlines()
      outJson = {}
      for idx, line in enumerate(jf['data']):
          origianlID = jf['data'][idx]['id']
          outJson[origianlID] = afLines[idx][:-2]
      json.dump(outJson, outF, indent=4)


  #----- evaluate all ------#
  inputQA = "./pipline_output/qa/test/predict_predictions.json"
  inputSum = "./pipline_output/sum/test_sum-pred.json"
  output = './pipline_output/test_all-pred.csv'

  with open(inputQA, 'r') as in1F, open(inputSum, 'r') as in2F, open(output, 'w') as outF:
    qaJson = json.load(in1F)
    sumJson = json.load(in2F)
    csvwriter = csv.writer(outF)
    csvwriter.writerow(['id', 'spoiler'])
    for i in range(400):
      keyStr = str(i)
      if keyStr in qaJson.keys():
        csvwriter.writerow([i, qaJson[keyStr]])
      elif keyStr in sumJson.keys():
        csvwriter.writerow([i, sumJson[keyStr]])


if __name__ == "__main__":
    main()