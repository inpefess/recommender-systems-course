# Copyright 2021-2022 Boris Shminke
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
"""
SLIM recommender
================

https://github.com/KarypisLab/SLIM
https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
from rs_datasets import MovieLens
from rs_metrics import hitrate

# pylint: disable=import-error
from SLIM import SLIM, SLIMatrix

from rs_course.utils import movielens_split


def slim_recommender(dataset_size: str) -> None:
    """
    >>> slim_recommender("small")
    Learning takes...
    0.55

    :param dataset_size: a size of MovieLens dataset to use
    """
    train, test, _ = movielens_split(
        MovieLens(dataset_size).ratings, 0.95, True
    )
    trainmat = SLIMatrix(train)
    model = SLIM()
    model.train({}, trainmat)
    model.save_model(modelfname="slim_model.csr", mapfname="slim_map.csr")
    testmat = SLIMatrix(train, model)
    slim_pred = model.predict(testmat, outfile="slim_recommendations.txt")
    pred = {int(k): list(map(int, v)) for k, v in slim_pred.items()}
    print(hitrate(test, pred))
