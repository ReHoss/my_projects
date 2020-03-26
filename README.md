## ResNet_Go
Residual Network model to learn policy and value in computer Go using Facebook ELF data.

- remy_go_example.py: the python file is an example of a model implementation and training. There exist more than 25 python files of this type. This one is a good representation of what is done for creating a model.

- RemGo8.h5: final model.

- rapport_Go_HOSSEINKHAN.pdf: the report of main experiments.


2020 - Hosseinkhan Boucher Rémy



Add details T. Cazenave
## HMM_PySpark
Implementation of the Viterbi algorithm (EM) for the estimation of parameters of Hidden Markov Model in a distributed fashion (using PySpark).

Implementation of the transition matrix and emission matrix estimation (Viterbi algorithm) algorithm from the book: [Data-Intensive Text Processing
with MapReduce][1](Jimmy Lin and Chris Dyer). It is a map-reduce based approach. Two distinct implementations are provided: one only using python built-in packages and replicating the book pseudo-code and one using the NumPy libraby and some optimizations.
See the report for a detailed description.

[1]: https://lintool.github.io/MapReduceAlgorithms/MapReduce-book-final.pdf

- hmm_python.py: built in python only implementation (no extra package needed). It intents to replicate the map-reduce based implementation from the reference book.

- hmm_numpy.py: numpy based, optimized implementation (recusrive forward-backward algorithm).

- hmm_report.pdf: report explaining model and implementation. Also contains performance comparison and commentaries on the book point of view.


2020 - Hosseinkhan Boucher Rémy



## Monte_Carlo_methods
Application of some Monte Carlo methods (e.g. Recycling rejected values in accept-reject methods, Metropolis–Hastings...) written in R.
Project written in R.

- project.pdf: stated problems (e.g. estimation using anthitetic variates, variance reduction).
- report_solutions.pdf: report containing solutions of different problems.
- solutions.R: script containing problem solutions.


2018 - Hosseinkhan Boucher Rémy



