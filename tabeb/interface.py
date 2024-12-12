from __future__ import annotations

import os
import numpy as np
from collections.abc import Sequence
from typing import Any, Protocol, Union, runtime_checkable
from sklearn.base import BaseEstimator, TransformerMixin

def get_model(model_name: str, revision: str | None = None, **kwargs: Any) -> TabEBBaseEncoder:
    """A function to fetch a model object by name.

    Args:
        model_name: Name of the model to fetch
        revision: Revision of the model to fetch
        **kwargs: Additional keyword arguments to pass to the model loader

    Returns:
        A model object
    """
    ## TODO: refacto
    if model_name == "random_encoder":
        from tabeb.models.random_encoder import RandomEncoder
        model = RandomEncoder(**kwargs)
    elif model_name == "carte_encoder":
        from tabeb.models.carte_encoder import CARTEEncoder
        model = CARTEEncoder(**kwargs)
    elif model_name == "skrub_encoder":
        from tabeb.models.skrub_encoder import SkrubEncoder
        model = SkrubEncoder(**kwargs)
    return model

class TabEBBaseEncoder(BaseEstimator, TransformerMixin):
    """The interface for an encoder in TabEB.

    """

    def __init__(self, device: str | None = None) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            device: The device to use for encoding. Can be ignored if the encoder is not using a device (e.g. for API)
        """
        self.name = "tabeb_base_encoder"

    def normalize(
        self, 
        embedding: Union[Sequence[str], np.ndarray]
    ) -> np.ndarray:
        """Normalizes the given NumPy array so that its norm is 1.

        Args:
            array: The NumPy array to normalize.

        Returns:
            The normalized NumPy array.
        """
        embedding = np.array(embedding)
        if embedding.ndim != 2:
            msg = (
            f"Expected 2D array, got {embedding.ndim}D array instead:\narray={embedding}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
            )
            raise ValueError(msg)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embedding / norms