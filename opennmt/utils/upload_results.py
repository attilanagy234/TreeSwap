import argparse
import json
import os
from datetime import datetime
from typing import Dict

import gspread
import yaml
from dotmap import DotMap


def main(config_path, status, run_folder_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    config = DotMap(config_yaml)

    if not isinstance(config.results.sheet_name, str):
        sheet_name = f'{config.general.src_postfix}-{config.general.tgt_postfix}'
    else:
        sheet_name = config.results.sheet_name

    gc = gspread.oauth()
    sh = gc.open_by_key(config.results.sheet_id)
    worksheet = sh.worksheet(sheet_name)

    if status == 'in_progress':
        row = len(worksheet.col_values(1)) + 1

        aug = config.augmentation
        aug_type = config.data.aug.path_src.split('/')[-1].split('_')[0] if isinstance(config.data.aug.path_src, str) else -1
        aug_sample = int(config.data.aug.path_src.split('/')[-3].split('-')[-2]) + 1 if isinstance(config.data.aug.path_src, str) else -1

        results = [config.general.src_postfix, config.general.tgt_postfix, aug.augmentation_ratio, aug_type,
                   aug.augmentation_type, aug.similarity_threshold, aug_sample, '-', '-',
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '', status, config_path]
        worksheet.update(f'A{row}:M{row}', [results])

    else:
        if status == 'done':
            with open(os.path.join(run_folder_path, 'final_result.txt'), 'r') as f:
                final_results: Dict = json.loads(f.read())
            results = [final_results['score'], final_results.get('meteor_score', -1)]
            run_folder_path = os.path.abspath(run_folder_path)
        else:
            results = ['-', '-']
            run_folder_path = ''


        configs, statuses = worksheet.batch_get(['M:M', 'L:L'], major_dimension='COLUMNS')
        rows = [i + 1 for i, (config, status) in enumerate(zip(configs[0], statuses[0])) if
                config == config_path and status == 'in_progress']

        batch_to_update = []

        if len(rows) != 1:
            # error, overwrite an empty row with the results
            row = len(configs[0]) + 1
            batch_to_update.extend([{'range': f'A{row}:G{row}', 'values': [['-' for _ in range(7)]]},
                                    {'range': f'M{row}', 'values': [[config_path]]}])

        else:
            row = rows[0]

        batch_to_update.extend([{'range': f'H{row}:I{row}', 'values': [results]},
                                {'range': f'K{row}:N{row}',
                                 'values': [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, config_path,
                                             run_folder_path]]}])
        worksheet.batch_update(batch_to_update)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='[REQUIRED] Path to the config file')
    parser.add_argument('--status', type=str, choices=['failed', 'in_progress', 'done'], required=True,
                        help='[REQUIRED] Training status')
    parser.add_argument('--run_folder', type=str, required=False, help='[REQUIRED] Path to the run folder.')

    args = parser.parse_args()
    main(args.config_path, args.status, args.run_folder)
