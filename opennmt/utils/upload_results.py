import argparse
import os
from datetime import datetime
import json

import gspread
import yaml
from dotmap import DotMap


def main(config_path, status, result_path):
    with open(config_path, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    config = DotMap(config_yaml)

    if config.results.sheet_name is not str:
        sheet_name = f'{config.general.src_postfix}-{config.general.tgt_postfix}'
    else:
        sheet_name = config.results.sheet_name

    gc = gspread.oauth()
    sh = gc.open_by_key(config.results.sheet_id)
    worksheet = sh.worksheet(sheet_name)

    if status == 'in_progress':
        row = len(worksheet.col_values(1)) + 1

        aug = config.augmentation
        aug_type = config.data.aug.path_src.split('/')[-1].split('_')[0]
        aug_sample = int(config.data.aug.path_src.split('/')[-3].split('-')[-2]) + 1

        results = [config.general.src_postfix, config.general.tgt_postfix, aug.augmentation_ratio, aug_type,
                   aug.augmentation_type, aug.similarity_threshold, status, '-', '-', os.path.abspath(config_path),
                   aug_sample, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        worksheet.update(f'A{row}:L{row}', [results])

    else:
        results = [status]
        if status == 'done':
            with open(result_path, 'r') as f:
                final_results = json.loads(f.read())
            results.extend([final_results['score'], final_results['meteor_score']])
        else:
            results.extend(['-', '-'])

        row = worksheet.find(os.path.abspath(config_path)).row
        worksheet.batch_update([{'range': f'G{row}:I{row}', 'values': [results]},
                                {'range': f'M{row}', 'values': [[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]]}])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='[REQUIRED] Path to the config file')
    parser.add_argument('--status', type=str, choices=['failed', 'in_progress', 'done'], required=True,
                        help='[REQUIRED] Training status')
    parser.add_argument('--result_path', type=str, required=False, help='[REQUIRED] Path to the final results.')

    args = parser.parse_args()
    main(args.config_path, args.status, args.result_path)
