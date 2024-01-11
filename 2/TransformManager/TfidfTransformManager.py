from sklearn.pipeline import FunctionTransformer, make_pipeline

from Document import Document
from TransformManager.TransformManager import TransformManager


class TfidfTransformManager(TransformManager):
    def __init__(self, vectorizer, pca, labels: list[str]):
        self.pipeline = make_pipeline(
            vectorizer,
            FunctionTransformer(lambda x: x.toarray(), accept_sparse=True),
            pca
        )
        super().__init__(labels)

    def _createNormlizeTexts(self, sentences: list[Document]):
        return [' '.join(text.tokens) for text in sentences]

    def _createTrainX(self, sentences: list[Document]):
        return self.pipeline.fit_transform(self._createNormlizeTexts(sentences))

    def _createTestX(self, sentences: list[Document]):
        return self.pipeline.transform(self._createNormlizeTexts(sentences))
