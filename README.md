## Conjugate gradient method analysis

Study of the preconditioned method and its floating point arithmetic properties (rounding error propagation).

- report.pdf: report of the case study.
- script_matlab.m: little MatLab test script.


2016 - Hosseinkhan Boucher Rémy



## Hidden Markov Model: Map-Reduce implementation of the Baum-Welch algorithm (EM) 

Implementation of the Baum-Welch algorithm (EM) for the estimation of parameters of Hidden Markov Model in a distributed fashion (using PySpark).

Implementation of the transition matrix and emission matrix estimation (forward-backward algorithm) algorithm from the book: [Data-Intensive Text Processing
with MapReduce][1](Jimmy Lin and Chris Dyer). It is a map-reduce based approach. Two distinct implementations are provided: one only using python built-in packages and replicating the book pseudo-code and one using the NumPy libraby and some optimizations.
See the report for a detailed description.

[1]: https://lintool.github.io/MapReduceAlgorithms/MapReduce-book-final.pdf

- hmm_python.py: built in python only implementation (no extra package needed). It intents to replicate the map-reduce based implementation from the reference book.

- hmm_numpy.py: numpy based, optimized implementation (recusrive forward-backward algorithm).

- hmm_report.pdf: report explaining model and implementation. Also contains performance comparison and commentaries on the book point of view.


2020 - Hosseinkhan Boucher Rémy



## Markov decision process: monte-carlo estimation, dynamic programming with toy models

Il a deux dossiers, pour les deux jeux.

Pour chaque dossier, les fonctions sont ecrites dans les scripts .py


Les notebooks .ipynb permettent de visualiser ce que produisent les scripts.
Dans chaque notebook on y importe nos scripts.
Les commentaires sont parfois ecrits en anglais ou francais, cela depend de l'auteur des lignes.

Les packages a installer sont :

plotly: https://plot.ly/python/
cufflinks: https://plot.ly/python/v3/ipython-notebooks/cufflinks/


Ces deux packages permettent de faire des graphes interactifs sur les notebook, veuillez nous contacter par mail en cas de quelconque souci.

Les notebooks sont aussi fourni en format .html


HOSSEINKHAN Remy / DESCHAMPS Theo


2019 - Hosseinkhan Boucher Rémy



## Computer Go: deep policy-value residual network

Residual Network model to learn policy and value in computer Go using Facebook ELF data.

- remy_go_example.py: the python file is an example of a model implementation and training. There exist more than 25 python files of this type. This one is a good representation of what is done for creating a model.

- RemGo8.h5: final model.

- rapport_Go_HOSSEINKHAN.pdf: the report of main experiments.



2020 - Hosseinkhan Boucher Rémy

Add details T. Cazenave



## Monte_Carlo_methods

Application of some Monte Carlo methods (e.g. Recycling rejected values in accept-reject methods, Metropolis–Hastings...) written in R.
Project written in R.

- project.pdf: stated problems (e.g. estimation using anthitetic variates, variance reduction).
- report_solutions.pdf: report containing solutions of different problems.
- solutions.R: script containing problem solutions.


2018 - Hosseinkhan Boucher Rémy

## Dupire local volatility model, calibration of a stochastic differential equation.

Ce projet traite de la calibration d une equation differentielle backward (solution terminale) i.e. la résolution de l´EDP de [Dupire][1] en utilisant son classique schéma de discrétisation associé: ´The Dupire equation is a partial differential equation (PDE) that links the contemporaneous prices of European call options of all strikes and maturities to the instantaneous volatility of the price process, assumed to be a function of price and time only.´

Implémentation du schéma de résolution numérique et calibration en Python (SciPy).


- Dupire2.py: est le script principal pour la création de la surface et la calibration.

- file_manip.py: permet d'extraire et de formater des données brut (csv, tsv...)

- 3DplotErrDupire.csv: est l'ensemble des points permettant de créer le graphe des erreurs selon beta1 et beta2.

- CalibrationDupire.pdf: est un fichier contenant quelques notes de travail.

- Presentation.pptx: est le diaporama présenté

Autres références:
[1]: https://www.lpsm.paris/pageperso/tankov/MA/poly_volatilite.pdf
https://www.rocq.inria.fr/mathfi/Premia/free-version/doc/premia-doc/pdf_html/fd_dupire_doc/index.html

2018 - Hosseinkhan Boucher Rémy


## Study of RLC circuit differential equations

Study of the second order differential equation that caracterized a non-linear [RLC][1].

[1]: https://en.wikipedia.org/wiki/RLC_circuit


- report.pdf: subject explanation and numerical illusrtations.


2017 - Hosseinkhan Boucher Rémy



