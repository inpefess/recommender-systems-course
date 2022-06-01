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
an integration testing script showing different algorithm performance
"""
import torch

if torch.cuda.is_available():
    print(f"CUDA device index: {torch.cuda.current_device()}")

# pylint: disable=wrong-import-position
from rs_datasets import MovieLens

# pylint: disable=wrong-import-position
from rs_course.cf_als import als_recommendations

# pylint: disable=wrong-import-position
from rs_course.cf_svd import pure_svd_recommender

# pylint: disable=wrong-import-position
from rs_course.cold_start import cold_start

# pylint: disable=wrong-import-position
from rs_course.collaborative_filtering_knn import collaborative_filtering_knn

# pylint: disable=wrong-import-position
from rs_course.content_based_knn import get_content_based_recommender

# pylint: disable=wrong-import-position
from rs_course.dnn_rs import dnn_recommender

# pylint: disable=wrong-import-position
from rs_course.lightfm_bpr import lightfm_recommender

# pylint: disable=wrong-import-position
from rs_course.popular import popular_recommender

movielens = MovieLens("25m")
popular_recommender(movielens.ratings, False)  # 0.5764079985323793
popular_recommender(movielens.ratings, True)  # 0.16636455186304128
get_content_based_recommender(movielens, 1)  # 0.21671701913393757
collaborative_filtering_knn(movielens.ratings, 20, 1)  # 0.4241691842900302
als_config = {"factors": 128, "use_gpu": True, "random_state": 0}
_, _, hitrate, _, _ = als_recommendations(movielens.ratings, als_config, 1)
print(hitrate)  # 0.4769385699899295
pure_svd_recommender(
    movielens.ratings, 2, {"n_components": 128, "random_state": 0}
)  # 0.48398791540785496
lightfm_recommender(
    movielens.ratings,
    {
        "no_components": 128,
        "loss": "bpr",
        "learning_rate": 0.01,
        "random_state": 0,
    },
    {"epochs": 15, "verbose": True},
    1,
)  # 0.3957703927492447
print(
    dnn_recommender(
        movielens.ratings,
        {
            "data_path": ".",
            "eval_step": 0,
            "epochs": 1,
            "train_batch_size": 2**13,
            "mf_embedding_size": 128,
            "mlp_embedding_size": 128,
            "mlp_hidden_size": [128],
            "use_gpu": True,
        },
    )
)  # ~0.05
cold_start(movielens, als_config, 1)  # 0.5166163141993958
