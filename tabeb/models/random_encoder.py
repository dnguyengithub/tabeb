import numpy as np
from tabeb.interface import TabEBBaseEncoder


class RandomEncoder(TabEBBaseEncoder):
    """
    A scikit-learn compatible encoder that generates random embeddings for input data.

    This encoder takes an array of shape [n_row, n_feature] as input and generates a random array of shape 
    [n_row, embedding_size] as output. The norm of each embedding is 1.

    Attributes:
        embedding_size (int): The size of the embedding vector.
        random_state (int | None): The seed for the random number generator.

    Methods:
        fit(X, y=None): Fits the encoder to the data (no-op for this encoder).
        transform(X): Transforms the input data into random embeddings.
    """

    def __init__(self, embedding_size: int = 128, random_state: int | None = None):
        """
        Initializes the RandomEncoder.

        Args:
            embedding_size (int): The size of the embedding vector.
            random_state (int | None): The seed for the random number generator.
        """
        super().__init__()
        self.name = "random_encoder"
        self.embedding_size = embedding_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the encoder to the data. This is a no-op for this encoder.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray, optional): The target values (ignored).

        Returns:
            RandomEncoder: The fitted encoder.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data into random embeddings.

        Args:
            X (np.ndarray): The input data of shape [n_row, n_feature].

        Returns:
            np.ndarray: The transformed data of shape [n_row, embedding_size] with each embedding normalized to have a norm of 1.
        """
        np.random.seed(self.random_state)
        n_row, n_feature = X.shape
        embeddings = np.random.randn(n_row, self.embedding_size)
        return self.normalize(embeddings)

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(10, 100)
    encoder = RandomEncoder(embedding_size=128, random_state=42)
    embeddings = encoder.fit_transform(X)
    print(embeddings.shape)  # Output: (10, 128)
    print(np.linalg.norm(embeddings, axis=1))  