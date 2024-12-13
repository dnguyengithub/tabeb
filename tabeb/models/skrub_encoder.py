import numpy as np
from skrub import TableVectorizer
from tabeb.interface import TabEBBaseEncoder


class SkrubEncoder(TabEBBaseEncoder):
    """
    """

    def __init__(self, **kwargs):
        """
        """
        super().__init__()
        self.name = "skrub_encoder"
        self.table_vectorizer = TableVectorizer(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        """
        return self.table_vectorizer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        """
        return np.nan_to_num(self.table_vectorizer.transform(X), 0.0)