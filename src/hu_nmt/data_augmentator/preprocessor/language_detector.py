import fasttext

PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'


class LanguageDetector:
    def __init__(self):
        self.model = fasttext.load_model(PRETRAINED_MODEL_PATH)

    def predict(self, text):
        # Extract ISO language code from model response
        return self.model.predict(text)[0][0].rpartition('__')[-1]

