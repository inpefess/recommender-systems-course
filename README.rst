============================
Building Recommender Systems
============================

Description
===========

A recommender system (RS) can help to influence your customers’
behaviour directly but entertainingly. In this course, we will build
RS’s using different approaches: content-based, collaborative
filtering, context-aware, or a hybrid one. We will learn about the
theory behind diverse mathematical models of an RS task: matrix and
tensor decompositions, associative rules, neighbourhood methods,
learning to rank, and metric learning. For the practical part, we
will employ classical machine learning (such as ``scikit-learn``),
deep learning (e.g. ``pytorch``), and a slew of specialised packages
(``implicit`` and ``lightfm`` amongs them). During the lectures, we
will talk not only about theorems but also about applications of RS’s
making the clients of companies and non-profit organisations happier.
No prior knowledge of the subject is necessary. Python programming
experience is mandatory. Statistical learning fundamentals will be
nice to have.

Topics
======

* Lecture 1. Introduction to the course. What is an RS? RS
  validation.
* Lecture 2. Content-based (CB) recommender systems. Classifiers,
  neighbourhood methods, item-to-item recommendations.
* Lecture 3. Collaborative filtering (CF). Associative rules,
  similarities with CB
* Lecture 4. Advanced CF methods. Matrix and tensor decompositions,
  factorisation machines, ALS and PureSVD
* Lecture 5. Recommendations in production. Learning to rank and
  multi-stages architectures
* Lecture 6. Deep learning in RS. Metric learning, two towers, deep
  and wide architecture
* Lecture 7. Cold-start problem. CB2CF, MaxVol
* Lecture 8. Advanced topics in RS: context-aware RS, factorization
  machines

How to Install
==============

On `Google Colab <https://colab.research.google.com/>`__ installation
takes 3-5 minutes:

.. code:: sh

   pip install git+https://github.com/inpefess/recommender-systems-course.git

On Linux. Installation process is the same as on Colab.

On Windows. Use `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about#main>`__
On WSL, proceed as on any Linux. Anaconda support is comming in next releases.

On macOS. Should be in principle installable in a similar way as on
Linux, but not tested.
