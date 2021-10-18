import argparse
import os


def pair_augmentation(path: str, augmentation_type: str, output_file: str):
    original_files = sorted(os.listdir(os.path.join(path, f'original_{augmentation_type}')))
    augmentation_files = sorted(os.listdir(os.path.join(path, augmentation_type)))

    with open(os.path.join(path, f'original_{augmentation_type}', original_files[0])) as original_file_1:
        with open(os.path.join(path, f'original_{augmentation_type}', original_files[1])) as original_file_2:
            with open(os.path.join(path, augmentation_type, augmentation_files[0])) as augmentation_file_1:
                with open(os.path.join(path, augmentation_type, augmentation_files[1])) as augmentation_file_2:
                    with open(output_file, 'w') as output_file:
                        while original_file_1_line := original_file_1.readline():
                            output_file.write('-' * 50 + '\n')
                            output_file.write('Original:\n')
                            output_file.write(original_file_1_line)
                            output_file.write(original_file_2.readline())
                            output_file.write(original_file_1.readline())
                            output_file.write(original_file_2.readline())
                            output_file.write('\nAugmented:\n')
                            output_file.write(augmentation_file_1.readline())
                            output_file.write(augmentation_file_2.readline())
                            output_file.write(augmentation_file_1.readline())
                            output_file.write(augmentation_file_2.readline())
                            output_file.write('-' * 50 + '\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='[REQUIRED] Path to basic augmentation output')
    parser.add_argument('--augmentation_type', type=str, help='[REQUIRED] Type of the augmentation we want to pair up')
    parser.add_argument('-o', '--output', type=str, default='augmentation_pairs.txt', help='[OPTIONAL] Output file path')
    args = parser.parse_args()

    pair_augmentation(args.path, args.augmentation_type, args.output)
