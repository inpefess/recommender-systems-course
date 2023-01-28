# Copyright 2023 Boris Shminke
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
A container class for RS training parameters
============================================

"""
from dataclasses import dataclass


@dataclass
class RSParams:
    """
    A collection of general recommender system parameters.

    :param split_test_users_into: into how many chunks to split the test set
    :param top_k: the number of items to recommend
    :param train_percentage: percentage of user-item pairs to leave in the
        training set
    :param warm_users_only: test on only those users, who were in training set
    """

    split_test_users_into: int
    top_k: int
    train_percentage: float
    warm_users_only: bool
