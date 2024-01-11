from abc import ABC, abstractmethod

from DocumentLoader import DocumentLoader


class TransformManager(ABC):
    _labels: list[str]

    def __init__(self, labels: list[str]):
        self._labels = labels

    def createTrainData(self, sentences: list[DocumentLoader]):
        return self._createTrainX(sentences), self._createY(sentences)

    def createTestData(self, sentences: list[DocumentLoader]):
        return self._createTestX(sentences), self._createY(sentences)

    @abstractmethod
    def _createTrainX(self, sentences: list[DocumentLoader]):
        raise NotImplementedError()

    @abstractmethod
    def _createTestX(self, sentences: list[DocumentLoader]):
        raise NotImplementedError()

    def _createY(self, sentences: list[DocumentLoader]):
        return [self._labels.index(text.text.label) for text in sentences]
