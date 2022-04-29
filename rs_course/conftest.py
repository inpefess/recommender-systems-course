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
from pytest import fixture
from rs_datasets import MovieLens

from rs_course.utils import filter_users_and_items


@fixture
def test_dataset() -> MovieLens:
    """
    :returns: a tiny MovieLens-like dataset for unit-tests
    """
    movielens = MovieLens("small")
    movielens.ratings = filter_users_and_items(
        MovieLens("small").ratings, 1000, 200
    )
    filtered_items = movielens.ratings["item_id"].unique()
    movielens.items = movielens.items[
        movielens.items["item_id"].isin(filtered_items)
    ]
    movielens.tags = movielens.tags[
        movielens.tags["item_id"].isin(filtered_items)
    ]
    return movielens
