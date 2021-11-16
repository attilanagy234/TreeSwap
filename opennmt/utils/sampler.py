from os import listdir, mkdir
from os.path import exists, isfile, join

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from opennmt.utils.logger import get_logger

log = get_logger(__name__)

HUNGLISH_PATH = '../../data/ftp.mokk.bme.hu/Hunglish2'
RANDOM_SEED = 500


class HunglishSampler:
    def __init__(self, base_data_dir, sample_from_domains, seed, samples_per_domain_in_valid,
                 samples_per_domain_in_test):
        self.domains = [
            'classic.lit',
            'law',
            'modern.lit',
            'softwaredocs',
            'subtitles'
        ]
        self.base_data_dir = base_data_dir
        self.sample_from_domains = sample_from_domains
        self.RANDOM_SEED = seed

        self.samples_per_domain_in_valid = samples_per_domain_in_valid
        self.samples_per_domain_in_test = samples_per_domain_in_test

    def sample(self):
        domain_data = {}
        data = {
            'train': {
                'hun': [],
                'eng': [],
                'source_file': [],
                'domain': []
            },
            'valid': {
                'hun': [],
                'eng': [],
                'source_file': [],
                'domain': []
            },
            'test': {
                'hun': [],
                'eng': [],
                'source_file': [],
                'domain': []
            }
        }

        for domain in self.sample_from_domains:
            if domain not in self.domains:
                raise ValueError(f'Cannot sample from domain {domain}')
            domain_data[domain] = {'hun': [], 'eng': [], 'source_file': []}

            domain_path = f'{self.base_data_dir}/{domain}/bi'
            files = [f for f in listdir(f'{domain_path}') if isfile(join(f'{domain_path}', f))]
            for file in files:
                with open(f'{domain_path}/{file}', 'r', encoding='latin2') as f:
                    hun_sentences = []
                    eng_sentences = []
                    malformed_lines = {}
                    for line in f:
                        try:
                            hun_sentence, eng_sentence = line.rstrip('\n').split('\t')
                            hun_sentences.append(hun_sentence)
                            eng_sentences.append(eng_sentence)
                            if len(hun_sentences) != eng_sentences:
                                raise ValueError(f'Hun-eng sentence pair has bad formatting')
                        except:
                            if domain not in malformed_lines:
                                malformed_lines[domain] = []
                            malformed_lines[domain].append((f'line: {line}', f'file: {file}'))

                    domain_data[domain]['hun'].extend(hun_sentences)
                    domain_data[domain]['eng'].extend(eng_sentences)
                    domain_data[domain]['source_file'].extend([file for _ in range(len(hun_sentences))])

            train_idxs, test_idxs = train_test_split(np.arange(len(domain_data[domain]['hun'])),
                                                     test_size=self.samples_per_domain_in_test,
                                                     random_state=self.RANDOM_SEED)
            train_idxs, valid_idxs = train_test_split(train_idxs,
                                                      test_size=self.samples_per_domain_in_valid,
                                                      random_state=self.RANDOM_SEED)

            for feature in ('hun', 'eng', 'source_file'):
                #                 Not memory efficient
                #                 data['train'][feature].extend(np.array(domain_data[domain][feature])[train_idxs].tolist())
                #                 data['valid'][feature].extend(np.array(domain_data[domain][feature])[valid_idxs].tolist())
                #                 data['test'][feature].extend(np.array(domain_data[domain][feature])[test_idxs].tolist())

                for idx in train_idxs:
                    data['train'][feature].append(domain_data[domain][feature][idx])
                for idx in valid_idxs:
                    data['valid'][feature].append(domain_data[domain][feature][idx])
                for idx in test_idxs:
                    data['test'][feature].append(domain_data[domain][feature][idx])

            data['train']['domain'].extend([domain] * train_idxs.shape[0])
            data['valid']['domain'].extend([domain] * valid_idxs.shape[0])
            data['test']['domain'].extend([domain] * test_idxs.shape[0])

        log.info('Train set length: {}'.format(len(data['train']['hun'])))
        log.info('Validation set length: {}'.format(len(data['valid']['hun'])))
        log.info('Test set length: {}'.format(len(data['test']['hun'])))
        log.info('--------TRAIN--------')
        log.info(data['train']['hun'][0:3])
        log.info(data['train']['eng'][0:3])
        log.info('--------VALID--------')
        log.info(data['valid']['hun'][0:3])
        log.info(data['valid']['eng'][0:3])
        log.info('--------TEST--------')
        log.info(data['test']['hun'][0:3])
        log.info(data['test']['eng'][0:3])

        # Dump splits to dataframes
        self.train_set = pd.DataFrame(data['train'])
        self.valid_set = pd.DataFrame(data['valid'])
        self.test_set = pd.DataFrame(data['test'])

    def create_data_set_files(self, path, base_file_name):
        file_name_beginning = join(path, base_file_name + '-')

        f = lambda set_name, language: file_name_beginning + set_name + '.' + language

        self.train_set['hun'][:].to_csv(f('train', 'hu'), header=None, index=None, sep='\t')
        self.train_set['eng'][:].to_csv(f('train', 'en'), header=None, index=None, sep='\t')

        self.valid_set['hun'][:].to_csv(f('valid', 'hu'), header=None, index=None, sep='\t')
        self.valid_set['eng'][:].to_csv(f('valid', 'en'), header=None, index=None, sep='\t')

        self.test_set['hun'][:].to_csv(f('test', 'hu'), header=None, index=None, sep='\t')
        self.test_set['eng'][:].to_csv(f('test', 'en'), header=None, index=None, sep='\t')


if __name__ == '__main__':
    sampler = HunglishSampler(
        base_data_dir=HUNGLISH_PATH,
        sample_from_domains=[
            'classic.lit',
            'law',
            'modern.lit',
            'softwaredocs',
            'subtitles'
        ],
        samples_per_domain_in_valid=5000,
        samples_per_domain_in_test=5000,
        seed=RANDOM_SEED
    )
    sampler.sample()

    combined_path = join(HUNGLISH_PATH, 'combined-en-hu')
    if not exists(combined_path):
        mkdir(combined_path)
    sampler.create_data_set_files(HUNGLISH_PATH, 'combined-en-hu/hunglish2')