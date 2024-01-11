from multiprocessing import freeze_support
from pathlib import Path

import joblib
from dask.distributed import Client
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Document import Document
from DocumentLoader import DocumentLoader
from TransformManager.TfidfTransformManager import TfidfTransformManager
from TransformManager.Word2VecTransformManager import Word2VecTransformManager

def getAccuracy(stat_train, stat_test, text_class):
    X_train, Y_train = text_class.createTrainData(stat_train)
    X_test, Y_test = text_class.createTestData(stat_test)
    svm_classifier = SVC(kernel='linear')
    with joblib.parallel_backend("dask", scatter=[X_train, Y_train]):
        svm_classifier.fit(X_train, Y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    return accuracy

if __name__ == '__main__':
    freeze_support()
    client = Client(processes=False)

    document_loader: DocumentLoader = DocumentLoader()
    sentences: list[Document] = document_loader.loadData(Path('data/news.txt.gz'))
    LABELS: list[str] = list(set([sentence.text.label for sentence in sentences]))
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    w2v = Word2Vec([sentence.tokens for sentence in train_sentences])
    transform_manager = Word2VecTransformManager(w2v.wv, LABELS)
    accuracy = getAccuracy(train_sentences, test_sentences, transform_manager)
    print(f'Accuracy: {accuracy:.2f}')

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    pca = PCA(n_components=100)
    transform_manager = TfidfTransformManager(vectorizer, pca, LABELS)
    accuracy = getAccuracy(train_sentences, test_sentences, transform_manager)
    print(f'Accuracy: {accuracy:.2f}')
