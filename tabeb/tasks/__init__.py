from __future__ import annotations

from importlib.metadata import version

import abc
from typing import Any, Protocol, Union, runtime_checkable
import os
import numpy as np
import random
import torch
from tabeb.interface import TabEBBaseEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import json


NUM_TRAIN = 16
NUM_MAX = 500 #1e9

@runtime_checkable
class TabEBBaseTask(Protocol):
    """TabEBBaseTask Class.

    Attributes:
        task (str): The type of task.
        metric (str): The metric to evaluate the task.
        data_name_list (list of str or None): The list of data names.
        data_dir (str): The directory where data is stored.
        seed (int): The random seed for reproducibility.
        data_loaded_ (bool): Indicates if the data has been loaded.
        metadata (dict): Metadata information.

    """
    def __init__(

        self, 
        task: str = "classification", ## TODO: create a list of tasks
        metric: str = "accuracy", ## TODO: create a list of metrics
        split: [str] = ["train", "test"], ## TODO: create a list of splits
        data_name_list: [str | None] =  [],  
        data_dir: str = "",
        seed: int = 42, 
        **kwargs: Any
    ):
        """Initializes the task with the specified parameters.

        Args:
            task (str, optional): The type of task to perform. Defaults to "classification".
            metric (str, optional): The metric to evaluate the task. Defaults to "accuracy".
            split (list of str, optional): The data splits to use. Defaults to ["train", "test"].
            data_name_list (list of str or None, optional): The list of data names. Defaults to an empty list.
            data_dir (str, optional): The directory where data is stored. Defaults to an empty string.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
            **kwargs (Any): Additional keyword arguments.
        """
        __metaclass__ = abc.ABCMeta

        self.task = task
        self.metric = metric
        self.data_name_list = data_name_list
        self.data_dir = data_dir
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        for key, value in kwargs.items():
            self.key = value
        self.data_loaded_ = False
        self.metadata = {}

    def get_head(self, **kwargs: Any) -> Any:
        """_summary_

        Returns:
            Any: _description_
        """
        pass
    
    @staticmethod
    def _load_data(data_name, data_dir):
        """Load the preprocessed data."""
        data_pd_dir = os.path.join(data_dir, data_name)
        data_pd = pd.read_parquet(os.path.join(data_pd_dir, f"{data_name}.parquet"))
        data_pd.fillna(value=np.nan, inplace=True)
        with open(os.path.join(data_pd_dir, f"{data_name}_metadata.json")) as f:
            data_metadata = json.load(f)
        return data_pd[:NUM_MAX], data_metadata
    
    @staticmethod
    def get_splits(data, data_metadata, num_train=NUM_TRAIN, random_state=42):
        """Set train/test split given the random state."""
        target_name = data_metadata["target_name"]
        if num_train is None:
            num_train = int(len(data) * 0.8)
        X = data.drop(columns=target_name)
        y = data[target_name]
        y = np.array(y)

        if data_metadata.get("repeated", False):
            entity_name = data_metadata["entity_name"]
        else:
            entity_name = np.arange(len(y))

        groups = np.array(data.groupby(entity_name).ngroup())
        num_groups = len(np.unique(groups))
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=int(num_groups - num_train),
            random_state=random_state,
        )
        idx_train, idx_test = next(iter(gss.split(X=y, groups=groups)))

        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        return X_train, X_test, y_train, y_test

    def load_data(self) -> Any:
        """_summary_

        Args:
            data_name (str): _description_
            data_dir (str, optional): _description_. Defaults to "".

        Returns:
            Any: _description_
        """
        self.data_dict = {}
        for data_name in self.data_name_list:
            self.data_dict[data_name] = self._load_data(data_name, self.data_dir)
        self.data_loaded_ = True

    @abc.abstractmethod
    def get_score(self, y_true: np.array, y_pred: np.array = None) -> float:
        """_summary_

        Args:
            y_true (np.array): _description_
            y_pred (np.array): _description_

        Returns:
            float: _description_
        """
        pass


    def get_head(self):
        """_summary_
        """
        pass 

    def evaluate(
        self, 
        models:  [TabEBBaseEncoder],
        split: str = "test",
        model_kwargs: dict = {},
        na_fill_value: Union[int, float] = 0,
        save_dir: str = "",
        **kwarg
    ) -> Any:
        """_summary_

        Args:
            model (TabEBBaseEncoder): _description_
            split (str, optional): _description_. Defaults to "test".
            model_kwargs (dict, optional): _description_. Defaults to {}.
            na_fill_value (Union[int, float], optional): _description_. Defaults to 0.
        Returns:
            Any: _description_
        """
        if not self.data_loaded_:
            self.load_data()
        scores = {}
        for model in models:
            scores[model.name] = {}
            for data_name, (data, data_metadata) in self.data_dict.items():
                X_train, X_test, y_train, y_test = self.get_splits(data, data_metadata)
                Embedding_train = model.fit_transform(X_train, y_train, **model_kwargs)
                Embedding_test = model.transform(X_test)
                Embedding_train = np.nan_to_num(Embedding_train, na_fill_value)
                Embedding_test = np.nan_to_num(Embedding_test, na_fill_value)
                head = self.get_head()
                head.fit(Embedding_train, y_train, **model_kwargs)
                if self.task in ["classification",]:
                    y_score = head.predict_proba(Embedding_test)
                    score = self.get_score(y_test, y_score)
                else:
                    y_pred = head.predict(Embedding_test)
                    score = self.get_score(y_test, y_pred)
                scores[model.name][data_name] = score
        # Create DataFrame
        df_scores = pd.DataFrame.from_dict(scores, orient='index')
        # Optional: reset index to make encoder names a column
        df_scores = df_scores.reset_index().rename(columns={'index': 'encoder'})
        df_scores[f'{self.task}_average'] = df_scores[self.data_name_list].mean(axis=1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        df_scores.to_parquet(os.path.join(save_dir, f"scores_{self.task}.parquet"))
        return df_scores

class TabEBRegressionTask(TabEBBaseTask):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.task = "regression"
        self.metric = "rmse"
        self.data_name_list = ["wine_pl", "wine_vivino_price"]

    def get_head(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return LinearRegression()
    
    def get_score(self, y_true: np.array, y_pred: np.array = None) -> float:
        """_summary_

        Args:
            y_true (np.array): _description_
            y_pred (np.array): _description_

        Returns:
            float: _description_
        """
        from sklearn.metrics import root_mean_squared_error
        return root_mean_squared_error(y_true, y_pred)
    
class TabEBClassificationTask(TabEBBaseTask):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.task = "classification"
        self.metric = "rmse"
        self.data_name_list = ["spotify",]

    def get_head(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return LogisticRegression()
    
    def get_score(self, y_true: np.array, y_score: np.array = None) -> float:
        """_summary_

        Args:
            y_true (np.array): _description_
            y_pred (np.array): _description_

        Returns:
            float: _description_
        """
        from sklearn.metrics import f1_score, roc_auc_score
        if len(np.unique(y_true)) >= 2:
            # Binary classification
            y_pred = np.argmax(y_score, axis=1)
            return roc_auc_score(y_true, y_pred)
        else:
            # Multi-class classification
            micro_roc_auc_ovr = roc_auc_score(
                y_test,
                y_score,
                multi_class="ovr",
                average="micro",
            )
            return micro_roc_auc_ovr