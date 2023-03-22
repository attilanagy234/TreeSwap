import argparse
import evaluate
import json


def compute_meteor(pred_file: str, test_file: str, result_file: str):
    meteor = evaluate.load('meteor')

    with open(pred_file) as f:
        predictions = f.readlines()

    with open(test_file) as f:
        references = f.readlines()

    results = meteor.compute(predictions=predictions, references=references)

    with open(result_file, 'r') as f:
        prev_results = json.loads(f.read())

    prev_results['meteor_score'] = results['meteor']

    with open(result_file, 'w') as f:
        json.dump(prev_results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred', type=str, required=True, help='[REQUIRED] Model prediction.')
    parser.add_argument('--test', type=str, required=True, help='[REQUIRED] File to evaluate against.')
    parser.add_argument('--result', type=str, required=True, help='[REQUIRED] Path to the final_result.txt')

    args = parser.parse_args()

    compute_meteor(args.pred, args.test, args.result)
