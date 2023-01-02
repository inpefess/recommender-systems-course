# Copyright 2022 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# noqa: D205, D400
"""
Basic tests for a recommender
==============================
"""
from abc import abstractmethod
from typing import List
from unittest import TestCase

import numpy as np
import pandas as pd
from rs_datasets import MovieLens


def _check_user_id_list(user_ids: List[int]) -> None:
    if not isinstance(user_ids, list):
        raise TypeError("user IDs must be a list")
    if not user_ids:
        raise TypeError("a list of user IDs must not be empty")
    if set(map(type, user_ids)) != {int}:
        raise TypeError("user IDs must be a list of integers")


def _check_training_data(training_data: pd.DataFrame) -> None:
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame")
    if not {"user_id", "item_id", "rating"}.issubset(
        set(training_data.columns)
    ):
        raise TypeError(
            "Training data must include the following columns: ",
            "user_id, item_id, rating",
        )
    if not np.issubdtype(
        training_data.user_id.dtype, np.integer  # type: ignore
    ) or not np.issubdtype(
        training_data.item_id.dtype, np.integer  # type: ignore
    ):
        raise TypeError("User ID and item ID must be of integer type")
    if not np.issubdtype(
        training_data.rating.dtype, np.floating  # type: ignore
    ):
        raise TypeError("Rating must be of floating point type")


class BaseRecommender:
    """An abstract base class for a recommender."""

    def fit(self, training_data: pd.DataFrame) -> None:
        """
        Train a model given the data.

        Set model training parameters by implementing the ``__init__`` method.
        Define the logic of model training by overloading the ``_fit`` method.


        :param training_data: a dataset of historical user-item interactions
        :raises TypeError: if arguments are malformed
        """
        _check_training_data(training_data)
        self._fit(training_data)

    @abstractmethod
    def _fit(self, training_data: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, user_ids: List[int], top_k: int) -> pd.DataFrame:
        """
        Give recommendations for a list of users.

        Define the prediction algorithm by overloading the ``_predict`` method.

        :param user_ids: a list of users in need of recommendations
        :param top_k: the number of items to recommend for each user
        :returns: a recommendations of items for users with estimated ratings
        :raises TypeError: if arguments are malformed
        """
        _check_user_id_list(user_ids)
        if not isinstance(top_k, int):
            raise TypeError("K must be an integer")
        if top_k <= 0:
            raise TypeError("K must be positive")
        return self._predict(user_ids, top_k)

    @abstractmethod
    def _predict(self, user_ids: List[int], top_k: int) -> pd.DataFrame:
        raise NotImplementedError


class DummyRecommender(BaseRecommender):
    """A non-working implementation of a recommender."""

    def __init__(self, dummy_parameter: int):
        """
        Set the only parameter.

        :param dummy_parameter: dummy integer parameter
        """
        self.dummy_parameter = dummy_parameter

    def _fit(self, training_data: pd.DataFrame) -> None:
        pass

    def _predict(self, user_ids: List[int], top_k: int) -> pd.DataFrame:
        # this line is deliberately nonsensical
        return self.dummy_parameter  # type: ignore


class RecommenderTest(TestCase):
    """Test fit and predict methods."""

    def setUp(self):
        """Create a dummy recommender for testing."""
        recommender = DummyRecommender(1)
        small_dataset = MovieLens("small").ratings
        recommender.fit(small_dataset)
        self.user_ids = [1, 2, 3]
        self.top_k = 10
        self.recommendations = recommender.predict(self.user_ids, self.top_k)

    def test_recommendations(self):
        """Test returned recommendations."""
        self.assertIsInstance(self.recommendations, pd.DataFrame)
        self.assertSetEqual(
            {"user_id", "item_id", "rating"}, set(self.recommendations.columns)
        )
        self.assertTrue(
            np.issubdtype(self.recommendations.user_id.dtype, np.integer)
        )
        self.assertTrue(
            np.issubdtype(self.recommendations.item_id.dtype, np.integer)
        )
        self.assertTrue(
            np.issubdtype(self.recommendations.rating.dtype, np.floating)
        )
        self.assertSetEqual(
            set(self.recommendations.user_id.tolist()), set(self.user_ids)
        )
        self.assertSetEqual(
            set(
                self.recommendations.groupby("user_id")
                .count()
                .item_id.tolist()
            ),
            {self.top_k},
        )
