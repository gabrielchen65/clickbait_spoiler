# Import the necessary classes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import html
import json
from tqdm import tqdm
import pdb
import torch

model_name = "t5-base"
model_path = "/kaggle/input/tuned-t5-base/checkpoint-24000"
tuned = "tuned"
trainVal = "train"
batch_size = 50
input = '/kaggle/input/task2-sum/{}_4summary.json'.format(trainVal)
outputFile = './{}_sum_{}_{}.json'.format(trainVal, model_name, tuned)
with open(input, 'r') as f:
    inputJson = json.load(f)
input_texts = ['']*len(inputJson["data"])
for i in range(len(inputJson["data"])):
    input_texts[i] = inputJson["data"][i]['article']

# Load the tokenizer and the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

for i in tqdm(range(0,len(inputJson["data"]),batch_size)):
    # Tokenize the input texts and generate the summaries for the entire batch
    #print("tokenizing..")
    inputs = tokenizer(input_texts[i:i+batch_size], return_tensors="pt", max_length=1024, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    #print("inferencing..")
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=1000)
    summary_texts = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    # Print the generated summaries
    #print("Generated Summaries:")
    for j, summary_text in enumerate(summary_texts):
        inputJson["data"][j+i]['context'] = summary_text
        # print(f"Summary {j + 1}:")
        # print(summary_text)
with open(outputFile, 'w') as f:
    json.dump(inputJson, f)