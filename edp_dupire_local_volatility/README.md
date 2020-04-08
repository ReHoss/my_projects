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

