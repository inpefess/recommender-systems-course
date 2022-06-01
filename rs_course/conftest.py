# Copyright 2021-2022 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fixtures for unit tests
=======================
"""
import torch

if torch.cuda.is_available():
    torch.cuda.current_device()
# pylint: disable=wrong-import-position
from pytest import fixture

# pylint: disable=wrong-import-position
from rs_datasets import MovieLens

# pylint: disable=wrong-import-position
from rs_course.utils import filter_users_and_items


def _get_test_dataset(
    min_items_per_user: int, min_users_per_item: int
) -> MovieLens:
    movielens = MovieLens("small")
    movielens.ratings = filter_users_and_items(
        MovieLens("small").ratings, min_items_per_user, min_users_per_item
    )
    filtered_items = movielens.ratings["item_id"].unique()
    movielens.items = movielens.items[
        movielens.items["item_id"].isin(filtered_items)
    ]
    movielens.tags = movielens.tags[
        movielens.tags["item_id"].isin(filtered_items)
    ]
    return movielens


@fixture(autouse=True, scope="session")
def test_dataset() -> MovieLens:
    """
    >>> getfixture("recbole_test_data")  # noqa: F821
    <...>

    :returns: a tiny MovieLens-like dataset for unit-tests
    """
    return _get_test_dataset(1000, 200)


@fixture(autouse=True, scope="session")
def recbole_test_data() -> MovieLens:
    """
    :returns: a tiny MovieLens-like dataset for ``recbole`` test
    """
    return _get_test_dataset(100, 100)
