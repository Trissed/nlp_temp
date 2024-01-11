import gzip
from pathlib import Path
from typing import Generator
from nltk.corpus import stopwords
from tqdm.contrib.concurrent import process_map
from yargy.token import Token
from yargy.tokenizer import MorphTokenizer, PUNCT
from Document import Document
from Text import Text

class DocumentLoader:
    stop_words: list[str] = stopwords.words("russian") + stopwords.words("english")
    TOKENIZER = MorphTokenizer()

    def loadData(self, path: Path):
        texts: list[Text] = self._loadInFile(path)
        sentences: list[Document] = process_map(self._normalize, texts, chunksize=100)
        return sentences

    def _loadInFile(self, path: Path):
        with gzip.open(str(path.resolve()), "rt", encoding="utf-8") as file:
            return [Text(*line.strip().split("\t")) for line in file]

    def _normalize(self, text: Text) -> Document:
        tokens: Generator[Token] = self.TOKENIZER(text.text)
        tokens = filter(lambda x: x.type is not PUNCT, tokens)
        tokens = [item.normalized for item in tokens]

        tokens = list(filter(lambda x: x not in self.stop_words, tokens))
        return Document(text, tokens)
