# Support Vector Machine Implementation


## Introduction

This module presents 2 different implementations of an SVM model :
* The first implementation uses scikit learn framework.
* The second is coded from scratch.

The implemented model is a SVM model with the following attributes:
* Sequential Minimal Optimization (SMO) algorithm.


## Motivation

The purpose of this exercice is to gain a better understanding of Support Vector Machines models, as they are "among the best off-the-shelf supervised learning algorithm". (Andrew Ng)

It is also a valuable exercice to practice programming in python language.

## Code structure

The code is structured as follow :
```
pySVM
├- docs/
│   └- svm_andrew_ng_cs229.pdf
├- data/
│   ├- titanic/
│   └- us_election/
├- library
│   ├- doityourself/
│   │   ├- params/
│   │   └- model.py
│   └- scikit_learn_SVC/
│       ├- params/
│       └- model.py
├- performance
│   └- confusion_matrix.py
├- .gitignore
├- evaluate.py
├- predict.py
├- prepare.py
├- README.md
├- requirements.txt
├- tools.py
└- train.py
```

The module has 4 main fucntionnalities :

* prepare.py - process and clean data.
* train.py - fit a model given a set of data.
* predict.py - perform a prediction given a previoulsy fitted model.
* evaluate.py - evaluate the performance a model.

## Installation

To use the different implementations, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyNaiveBayes.git
```

### Using a virtual environment

First create the virtual environment :

```
$ python3 -m venv path_to_the_env
```

Activate it :

```
$ source path_to_the_env/bin/activate
```

Then install all the requirements :

```
$ pip install -r requirements.txt
```

## Test

To test if all the functionnalities are working :

```
$ python -m unittest discover -s unittest
```

## Use

For training using a file :

```
>>> form train import main
>>> TODO
```

Or from the terminal :

```
$ python train.py
```

## Author

Rémi Moise

moise.remi@gmail.com

## Licence

MIT License

Copyright (c) 2018
