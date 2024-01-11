import numpy as np
from gensim.models import KeyedVectors

from DocumentLoader import DocumentLoader
from TransformManager.TransformManager import TransformManager


class Word2VecTransformManager(TransformManager):
    def __init__(self, wv: KeyedVectors, labels: list[str]):
        self._wv: KeyedVectors = wv
        super().__init__(labels)

    def _createX(self, sentences: list[DocumentLoader]):
        results = []
        for text in sentences:
            vecs = []
            for word in text.tokens:
                try:
                    vecs.append(self._wv[word])
                except KeyError:
                    pass

            combined_vector = np.row_stack(vecs)
            mean_value = np.mean(combined_vector, axis=0)
            results.append(mean_value)

        return results

    def _createTrainX(self, sentences: list[DocumentLoader]):
        return self._createX(sentences)

    def _createTestX(self, sentences: list[DocumentLoader]):
        return self._createX(sentences)
