#!/usr/bin/python3
import argparse
import json
import csv
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def run_baseline(input_file, output_file, concat_multi=True):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        inp = list(inp)
        squad_like_dataset = {"data": []}
        count = 0
        for line in inp:             
            output = json.loads(line)

            #if output["tags"] != ["multi"]:
            #    continue
            question = str(output['targetTitle'])
            # concat context
            context = ''
            id = output['id'] if 'train' not in output_file else output['uuid']
            for i in output['targetParagraphs']:
                context += i
            # find answers         
            if 'test' in output_file:
                  answers = ['']
                  answer_start = [-1]
            else:
              #answers = [output['spoiler'][0]]
              #answer_start = [context.find(answers[0])]
              if len(output['spoiler']) == 1:
                  answers = output['spoiler']
                  answer_start = [context.find(answers[0])]
              else:
                  if not concat_multi:
                      answer_start = [-1]*len(output['spoiler'])
                      answers = output['spoiler']
                      for i, ele in enumerate(output['spoiler']):
                          # find answer_start location
                          answer_start[i] = context.find(ele)
                  else:
                      answers = [" ".join(output['spoiler'])]
                      answer_start = [context.find(answers[0])]                    
            # Create the data structure for a single SQuAD-like data point
            data_point = {
                "context": context,
                "question": question,
                "id": id,
                "answers": {"text": answers, "answer_start": answer_start}
            }

            # Append the data point to the dataset
            squad_like_dataset["data"].append(data_point)
            count += 1
            #if count > 9: break
        #squad_like_dataset = squad_like_dataset["data"]
        # Save the SQuAD-like dataset as a JSON file
        json.dump(squad_like_dataset, out)
            
           
if __name__ == '__main__':
    #args = parse_args()
    #run_baseline(args.input, args.output)
    #run_baseline('./train.jsonl', './train.json')
    run_baseline('./datasets/val.jsonl', './datasets/val-concat.json', True)
    #run_baseline('./test.jsonl', './test.json')

