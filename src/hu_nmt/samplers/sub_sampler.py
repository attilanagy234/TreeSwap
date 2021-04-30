class SubSampler:
    """
    Receives the entire dataset and returns a subsample to augment
    based on predefined criteria
    """
    def __init__(self, config):
        self._config = config

    def filter_by_length(self, df, token_length):
        df = self.add_length_fields_to_df(df)
        df = df[df['token_length'] < token_length]
        return df

    @staticmethod
    def add_length_fields_to_df(df):
        df['char_length'] = df['sentence'].apply(len)
        df['token_length'] = df['sentence'].apply(lambda x: len(x.split(' ')))
        return df
