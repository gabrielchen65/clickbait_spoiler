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


def predict(inputs):
    for i in inputs:
        #yield {'uuid': i['postId'], 'targetTitle': i['targetTitle'], 'spoiler':i['spoiler'], 'targetParagraphs':i['targetParagraphs'], 'id': i['id']}
        yield {'uuid': i['postId'], 'targetTitle': i['targetTitle'], 'spoiler':i['spoiler'], 'targetParagraphs':i['targetParagraphs']}


def run_baseline(input_file, output_file):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        #out.write('id,question,answer,answer_start,context\n')
        csvwriter = csv.writer(out, delimiter=',')
        csvwriter.writerow(['id','question','answers','answer_start','context'])
        count = 0
        for idx, output in enumerate(predict(inp)):
            question = str(output['targetTitle'])
            answers = str(output['spoiler'][0])
            context = ''
            for i in output['targetParagraphs']:
                context += i
            # find answer_start location
            answer_start = str(context.find(answers))
            csvwriter.writerow([str(idx), question, answers, answer_start, context])
            #out.write(str(output['id']) + ',"' + question + '","' + answer
            #          + '","' + answer_start + '","' + context + '"\n')
            count += 1
            #out.write(json.dumps(output) + '\n')
            #if ',' in output['spoiler']:
            #  out.write(str(output['id']) + ',' + str(output['targetTitle']) + ',"' + str(output['spoiler']) + '",' + str(output['targetParagraphs']) + '\n')
            #else:
            #  out.write(str(output['id']) + ',' + str(output['targetTitle']) + ',' + str(output['spoiler']) + ',' + str(output['targetParagraphs']) + '\n')


if __name__ == '__main__':
    #args = parse_args()
    #run_baseline(args.input, args.output)
    run_baseline('./datasets/train.jsonl', './datasets/train.csv')
    run_baseline('./datasets/val.jsonl', './datasets/val.csv')
    #run_baseline('./datasets/test.jsonl', './datasets/test.csv')

