from abc import ABC

from RAKE import Rake
from yake import yake


class AbstractKeywordExtraction(ABC):
    """
    Abstract class for keyword extraction

    """

    def __init__(self):
        self._name = 'AbstractKeywordExtraction'
        self.model = None

    def get_keywords(self, text: str) -> list:
        pass

    @property
    def name(self):
        return self._name


class RAKEKeywordExtraction(AbstractKeywordExtraction):
    """
    Class for keyword extraction using RAKE.
    """

    def __init__(self, stopwords_path=None, **kwargs):
        super().__init__()
        self._name = f'rake'
        self.model = Rake(stopwords_path)
        self.kwargs = kwargs

    def get_keywords(self, text: str) -> list:
        """
        Returns the keywords of the text.
        :param text:
        :return:
        """
        return self.model.run(text, **self.kwargs)


class YAKEKeywordExtraction(AbstractKeywordExtraction):
    """
    Class for keyword extraction using YAKE.
    """

    def __init__(self, stopwords_path=None, min_characters=3, **kwargs):
        super().__init__()
        self._name = f'yake'
        self.stopwords = self.load_stopwords(stopwords_path)
        self.model = yake.KeywordExtractor(stopwords=self.stopwords, **kwargs)
        self.min_chars = min_characters

    def get_keywords(self, text: str) -> list:
        """
        Returns the keywords of the text.
        :param text:
        :return:
        """
        return self.model.extract_keywords(text)

    def load_stopwords(self, path):
        with open(path, 'rt') as f:
            stopwords = f.read().splitlines()
        return stopwords
