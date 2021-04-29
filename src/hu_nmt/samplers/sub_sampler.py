class SubSampler:
    """
    Receives the entire dataset and returns a subsample to augment
    based on predefined criteria
    """
    def __init__(self, df, config):
        self._df = df
        self._config = config

    def subsample(self):

        raise NotImplementedError()

    @staticmethod
    def add_length_fields_to_df(df):
        df['char_length'] = df['sentence'].apply(len)
        df['token_length'] = df['sentence'].apply(lambda x: len(x.split(' ')))
        return df
