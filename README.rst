..
  Copyright 2021-2022 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

============================
Building Recommender Systems
============================

|CircleCI|\ |AppveyorCI|\ |Documentation Status|\ |codecov|\ |Zenodo|

This is a utility package for the course `SMEMI315: Building
Recommender systems
<https://syllabus.univ-cotedazur.fr/fr/course/router-light/SMEMI315>`__
taught at the Université Côte d'Azure in autumns of 2021 and 2022.
See the `course description
<https://recommender-systems-course.rtfd.io/en/latest/course-desc.html>`__
for more info about its content.

How to Install
==============

In a Docker container
----------------------

First, install `Docker <https://docs.docker.com/get-docker/>`__ on your system.

.. code:: sh

   docker build -t recommender-systems-course https://github.com/inpefess/recommender-systems-course.git
   docker run -it --rm -p 8888:8888 recommender-systems-course jupyter-lab --ip=0.0.0.0 --port=8888

On Linux
---------

This includes `Google Colab <https://colab.research.google.com/>`__
and `Kaggle <https://www.kaggle.com/docs/notebooks>`__.

.. code:: sh

   pip install git+https://github.com/inpefess/recommender-systems-course.git


On Windows
-----------

There are two options:

#. Use `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about#main>`__
   
   On WSL, proceed as on any Linux.

#. Use `Anaconda <https://conda.io/en/latest/miniconda.html>`__

   .. code:: sh

      # get the source
      git clone git+https://github.com/inpefess/recommender-systems-course.git
      cd recommender-systems-course
      # use a provided environment configuration
      conda env create -n recommender-systems-course -f environment.yml python=3.8
      conda activate recommender-systems-course
      # test that all installed correctly
      pytest
      # add ``rs_course`` package
      pip install .
      # start working
      jupyter lab

On macOS
---------

Should be in principle installable in a similar way as on
Linux, but not tested.

How to Use
===========

This package is supposed to be used together with ``rs_datasets``:

.. code:: python

    from rs_datasets import MovieLens

    ratings = MovieLens("small").ratings

The package contains pre-packed examples of different recommenders.
For example, this function computes ``hit-rate@1`` of a PureSVD
trained with default parameteres from ``scikit-learn`` on randomly
selected 80% of ratings:

.. code:: python

    from rs_course.cf_svd import
    pure_svd_recommender
    
    pure_svd_recommender(
	  ratings=ratings,
	  split_test_users_into=1,
	  model_config={},
	  top_k=1,
	  train_percentage=0.8
    )

More Detailed Documentation
============================

More detailed documentation is available `here
<https://recommender-systems-course.rtfd.io>`__.

Similar packages
=================

This package is not supposed to be used as a recommender systems
library. It's only purpose is to help a complete beginner to get the
taste of the recommenders' world. For a proper library, try something
from `this list <https://github.com/Darel13712/recsys_libraries>`__.

How to Cite
============

If you want to cite this package in your research paper, please use the following `DOI <https://doi.org/10.5281/zenodo.7096595>`__.

.. |CircleCI| image:: https://circleci.com/gh/inpefess/recommender-systems-course.svg?style=svg
   :target: https://circleci.com/gh/inpefess/recommender-systems-course
.. |Documentation Status| image:: https://readthedocs.org/projects/recommender-systems-course/badge/?version=latest
   :target: https://recommender-systems-course.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/inpefess/recommender-systems-course/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/inpefess/recommender-systems-course
.. |AppveyorCI| image:: https://ci.appveyor.com/api/projects/status/32ws0aamvby6mc6o?svg=true
   :target: https://ci.appveyor.com/project/inpefess/recommender-systems-course
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7096596.svg
   :target: https://doi.org/10.5281/zenodo.7096595
