import fire
import json
import os

def main(data_path):
    data_lists = list(open(data_path))
    samples = []
    for data in data_lists:
        data = json.loads(data)['code']
        if '###Instruction.' in data:
            data = data.replace('###Instruction.', '###Instruction:')
        if 'Output:' in data:
            samples.append({'instruction':data.split('Output:')[0].split('Instruction:')[1].split('###')[0],
                        'input': '',
                        'output': data.split('Output:')[1]})
    file_name = data_path.split('/')[-1].split('.')[0]
    path_dir = data_path.rsplit('/', 1)[0]
    with open(os.path.join(path_dir, file_name+'_formatted.json'), 'w') as f:
        json.dump(samples, f)


if __name__ == '__main__':
    fire.Fire(main)