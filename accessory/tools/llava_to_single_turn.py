import json
import os
import tqdm
import copy

json_path = "llava_instruct_150k.json"
datas = json.load(open(json_path))

out_datas = []
for data in tqdm.tqdm(datas):
    conversations = data['conversations']
    for i in range(len(conversations)//2):
        out_data = copy.deepcopy(data)
        del out_data['conversations']
        out_conversations = [{}, {}]
        for k,v in conversations[i*2].items():
            out_conversations[0][k] = v.replace('<image>', '').replace('\n', '')
        for k,v in conversations[i*2+1].items():
            out_conversations[1][k] = v.replace('<image>', '').replace('\n', '')
        out_data['conversations'] = out_conversations
        out_datas.append(out_data)

with open('llava_instruct_150k_single_turn.json', 'w') as f:
    json.dump(out_datas, f)