import os


def save_result(args, info, prompt, global_config, ds_collections, result_path='', dataset=''):
    os.makedirs('results', exist_ok=True)
    if isinstance(args.pretrained_path, list):
        pre_path = args.pretrained_path[0]
    else:
        pre_path = args.pretrained_path
    with open(f'results/{pre_path.split("ckpts")[-1].replace("/", "_")}/results.txt', 'a') as f:
        f.write('-' * 10 + '\n')
        f.write(f'{dataset}\n')
        f.write(f'generation config: {global_config}\n')
        f.write(f'dataset config: {ds_collections[dataset]}\n')
        f.write(f'used prompt: {prompt}\n')
        f.write(f'pretrained_path: {args.pretrained_path}\n')
        if result_path:
            f.write(f'result saved to {result_path}\n')
        f.write(str(info) + '\n')
        f.write('-' * 10 + '\n')
    print(f'pre_path: {pre_path}')
    print(f'dataset: {ds_collections[dataset]}')