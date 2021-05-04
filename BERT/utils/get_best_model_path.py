import os
import re
import argparse


def get_best_model_path(run_dir: str):
    files = os.listdir(run_dir)
    checkpoint_files = [f for f in files if f.endswith('.pt')]
    checkpoint_steps = [int(re.search(r'\d+', f).group()) for f in checkpoint_files]
    checkpoint_steps.sort()
    best_step = str(checkpoint_steps[-21])
    for file in files:
        if best_step in file:
            return os.path.join(run_dir, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, help='[OPTIONAL] Path of the directory where the model checkpoint files files are located.')


    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = '.'

    print(get_best_model_path(run_dir))

