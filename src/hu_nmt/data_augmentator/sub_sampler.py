class SubSampler:
    """
    Receives the entire dataset and returns a subsample to augment
    based on predefined criteria
    """
    def __init__(self, df):
        self._df = df

    def subsample(self):
        raise NotImplementedError()
