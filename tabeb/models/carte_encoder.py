import numpy as np
from tabeb.interface import TabEBBaseEncoder
from carte_ai import Table2GraphTransformer
from sklearn.preprocessing import PowerTransformer
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch_geometric.data import Batch

class CARTEBaseEncoder(nn.Module):
    def __init__(self, carte_estimator):
        super().__init__()

        # Copy all layers except the last Linear layer in ft_classifier
        self.ft_base = carte_estimator.ft_base

        # Recreate ft_classifier without the last Linear layer
        self.head = nn.Sequential(
            carte_estimator.ft_classifier[0],  # First Linear(300, 150)
            carte_estimator.ft_classifier[1],  # ReLU()
            carte_estimator.ft_classifier[2],  # LayerNorm(150)
            carte_estimator.ft_classifier[3],  # Second Linear(150, 75)
            carte_estimator.ft_classifier[4],  # ReLU()
            carte_estimator.ft_classifier[5]   # LayerNorm(75)
        )
       
    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x.clone(),
            input.edge_index.clone(),
            input.edge_attr.clone(),
            input.ptr[:-1],
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]
        x = self.head(x)

        return x

class CARTEEncoder(TabEBBaseEncoder):
    """
    A scikit-learn compatible encoder that using carte_ai (https://pypi.org/project/carte-ai/).

    """

    def __init__(
            self, 
            pretrain_model_path: str | None = None,
            lm_path: str | None = None,
            random_state: int | None = 0,
            num_models: int = 1,
            device: str = "cpu",
            n_jobs: int = 1,
            disable_pbar: bool = False,
            task: str = "regression", ## TODO: define the list of tasks
            **kwargs,
        ):
        """
        Initializes the CARTEEncoder.

        Args:
            pretrain_model_path (str | None): Path to the pretrained model. Must be specified.
            lm_path (str | None): Path to the language model. If None, downloads fastText model.
            random_state (int | None): Seed for random number generation. Default is 0.
            num_models (int): Number of models to use. Default is 1.
            device (str): Device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            n_jobs (int): Number of parallel jobs to run. Default is 1.
            disable_pbar (bool): Whether to disable progress bar. Default is False.
            task (str): Task type, e.g., 'regression'. Default is 'regression'.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If pretrain_model_path is not specified.

        """
        super().__init__()
        self.name = "carte_encoder"
        self.pretrain_model_path = pretrain_model_path
        self.random_state = random_state
        if lm_path is None:
            lm_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin") # download fastText
        self.lm_path = lm_path
        self.num_models = num_models
        self.device = device
        self.n_jobs = n_jobs
        self.disable_pbar = disable_pbar
        self.task = task
        for key, value in kwargs.items():
            self.key = value

        self.preprocessor = Table2GraphTransformer(fasttext_model_path=self.lm_path)


    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fits the CARTE model to the provided data.
        Args
            X : np.ndarray
                The input data to fit the model.
            y : np.ndarray, optional
                The target values (default is None).
        Returns:
            self : Returns the instance itself.
        """
        self.is_fitted_ = False
        if self.task == "regression":
            from carte_ai import CARTERegressor
            self.carte_estimator = CARTERegressor(
                num_model=self.num_models,
                random_state=self.random_state,
                device=self.device,
                n_jobs=self.n_jobs, 
                disable_pbar=self.disable_pbar,
            )
        elif self.task == "classification":
            from carte_ai import CARTEClassifier
            self.carte_estimator = CARTEClassifier(
                num_model=self.num_models,
                random_state=self.random_state,
                device=self.device,
                n_jobs=self.n_jobs, 
                disable_pbar=self.disable_pbar,
            )
        # Preprocess the data
        X_train = self.preprocessor.fit_transform(X_train, y=y_train)
        # Fit the model
        self.carte_estimator.fit(X_train, y_train)
        self.carte_estimator = self.carte_estimator.model_list_[0] # TODO: use bagging in the next version.
        self.encoder = CARTEBaseEncoder(self.carte_estimator)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data using the preprocessor and encoder.
        Args:
            X (np.ndarray): Input data to be transformed.
        Returns:
            np.ndarray: Transformed data embeddings.
        """

        # Preprocess the data
        X = self.preprocessor.transform(X)

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)

        # Transform
        with torch.no_grad():
            embedding = self.encoder(ds_predict_eval).cpu().detach().numpy()
        return embedding
    
    def _set_data_eval(self, data):
        """Constructs the aggregated graph object from the list of data.

        This is consistent with the graph object from torch_geometric.
        Returns the aggregated graph object.
        """
        make_batch = Batch()
        with torch.no_grad():
            ds_eval = make_batch.from_data_list(data, follow_batch=["edge_index"])
            ds_eval.to(self.device)
        return ds_eval
