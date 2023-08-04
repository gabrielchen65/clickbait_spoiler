import json
import pdb

input = './datasets/task2/val.jsonl'
extension = '.jsonl'
answerType = 'non-multi' # multi or non-multi
concat = False if answerType == 'non-multi' else True
if not concat:
  out_ext = '_4summary_phrase-passage.json'
else:
  out_ext = '_4summary_all-concat.json'
output = input.replace(extension, out_ext)

keepAttribute = ['uuid', 'targetParagraphs', 'targetTitle', 'spoiler','tags']
with open(input, 'r') as inF, open(output, 'w') as outF:
    jsonList = list(inF)
    jsonF = {"data":[]}
    for line in jsonList:
        jsonLine = json.loads(line)

        if answerType == 'non-multi':
          if jsonLine['tags'] == ['multi']: continue ####################
        elif answerType == 'multi':
          if jsonLine['tags'] != ['multi']: continue 
        # data = {}
        title = jsonLine['targetTitle'] if jsonLine['targetTitle'].endswith('?') else jsonLine['targetTitle']+':'
        paragraphs = ''
        for i in jsonLine['targetParagraphs']:
           paragraphs += i
        # data['article'] = title + ' ' + paragraphs
        # data['highlights'] = jsonLine['spoiler'] if 'test.jsonl' not in input else ""
        # data['id'] = jsonLine['uuid'] if 'train.jsonl' in input else jsonLine['id']
        # data['context'] = paragraphs
        # data['answers'] = jsonLine['spoiler'] if 'test.jsonl' not in input else ""
        # data['qeustion'] = jsonLine['targetTitle']

        article = title + ' ' + paragraphs
        #highlights = jsonLine['spoiler'] if 'test.jsonl' not in input else ""
        id = jsonLine['uuid'] if 'train.jsonl' in input else jsonLine['id']
        context = paragraphs
        question = jsonLine['targetTitle']

        if not concat:
          answers = jsonLine['spoiler'][0] if 'test.jsonl' not in input else "TRIVIAL"
          answer_start = context.find(answers)
        else:
          answers = " ".join(jsonLine['spoiler']) if 'test.jsonl' not in input else "TRIVIAL"
          answer_start = -1          

        # Create the data structure for a single SQuAD-like data point
        data_point = {
            "article" : article,
            "highlights" : answers,
            "context": paragraphs,
            "question": question,
            "id": id,
            "answers": {"text": [answers], "answer_start": [answer_start]},
            "tags": jsonLine['tags'] if 'test.jsonl' not in input else ["TRIVIAL"]
        }

        jsonF['data'].append(data_point)
    json.dump(jsonF, outF, indent=4)


###    data['article'] = title + ' ' + jsonLine['targetParagraphs']
###                      ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### TypeError: can only concatenate str (not "list") to str
###  CCCCCCCCould be a problem for previous training data
# (Pdb) jsonLine['targetParagraphs']
# ['It’ll be just like old times this weekend for Tom Brady and Wes Welker.', 'Welker revealed Friday morning on a Miami radio station that he contacted Brady because he’ll be in town for Sunday’s game between the New England Patriots and Miami Dolphins at Gillette Stadium. It seemed like a perfect opportunity for the two to catch up.', 'But Brady’s definition of "catching up" involves far more than just a meal. In fact, it involves some literal "catching" as the Patriots quarterback looks to stay sharp during his four-game Deflategate suspension.', '"I hit him up to do dinner Saturday night. He’s like, ‘I’m going to be flying in from Ann Arbor later (after the Michigan-Colorado football game), but how about that morning we go throw?’ " Welker said on WQAM, per The Boston Globe. "And I’m just sitting there, I’m like, ‘I was just thinking about dinner, but yeah, sure. I’ll get over there early and we can throw a little bit.’ "', 'Welker was one of Brady’s favorite targets for six seasons from 2007 to 2012. It’s understandable him and Brady want to meet with both being in the same area. But Brady typically is all business during football season. Welker probably should have known what he was getting into when reaching out to his buddy.', '"That’s the only thing we really have planned," Welker said of his upcoming workout with Brady. "It’s just funny. I’m sitting there trying to have dinner. ‘Hey, get your ass up here and let’s go throw.’ I’m like, ‘Aw jeez, man.’ He’s going to have me running like 2-minute drills in his backyard or something."', 'Maybe Brady will put a good word in for Welker down in Foxboro if the former Patriots wide receiver impresses him enough.']
# (Pdb) len(jsonLine['targetParagraphs'])