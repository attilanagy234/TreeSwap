import unittest

from hu_nmt.data_augmentator.preprocessor.language_detector import LanguageDetector


class LanguageDetectorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.langdetect = LanguageDetector()

    def testHungarianText(self):
        text = 'Ez egy gyönyörű ház'
        self.assertEqual('hu', self.langdetect.predict(text))

    def testEnglishText(self):
        text = 'This is a beautiful house'
        self.assertEqual('en', self.langdetect.predict(text))




