# Learning Operators with Coupled Attention

![Fig2 resized](https://user-images.githubusercontent.com/24652388/182936051-613aaa15-e743-4d7c-9ff6-e2d4093750fd.png)

Code and data accompanying the manuscript titled "Learning Operators with Coupled Attention", authored by Georgios Kissas*, Jacob H. Seidman*,  Leonardo Ferreira Guilhoto, Victor M. Preciado, George J.Pappas and Paris Perdikaris.
 
\* These authors contributed equally.

# Abstract

Supervised operator learning is an emerging machine learning paradigm with applications to modeling the evolution maps of spatio-temporal dynamical systems and approximating general black-box relationships between functional data. We propose a novel operator learning method, LOCA (Learning Operators with Coupled Attention), motivated from the attention mechanism. The input functions are mapped to a finite set of features which are then averaged with attention weights that depend on the output query locations. By coupling these attention weights together with an integral transform, LOCA is able explicitly learn correlations in the target output functions, enabling us to approximate nonlinear operators even when the number of output function measurements is very small. Our formulation is accompanied by rigorous approximation theoretic guarantees on the expressiveness of the proposed model. Empirically, we evaluate the performance of LOCA on several operator learning scenarios involving systems governed by ordinary and partial differential equations, as well as a black-box climate prediction problem. Through these scenarios we demonstrate state of the art accuracy, robustness with respect to noisy input data, and a consistently small spread of errors over testing data sets, even for out-of-distribution prediction tasks.


# Citation

    @article{JMLR:v23:21-1521, 
        author  = {Georgios Kissas and Jacob H. Seidman and Leonardo Ferreira Guilhoto and Victor M. Preciado and George J. Pappas and Paris Perdikaris},
         title   = {Learning Operators with Coupled Attention},
        journal = {Journal of Machine Learning Research},
        year    = {2022},
        volume  = {23},
        number  = {215},
        pages   = {1--63},
        url     = {http://jmlr.org/papers/v23/21-1521.html}
      }


The repository contains all the necassary code and data to reproduce the results in the paper. 

You can find a LOCA tutorial with explanation for the Darcy flow example [here](https://colab.research.google.com/drive/1axxLGhgwipCSw9WQVMBklvQdW_K99E1D?usp=sharing).

The training and testing data sets accompanying the manuscript can be found [here]() and the codes to plot the results as well as the data to reproduce the figures in the manuscript can be found [here](). 

You can find the codes for LOCA, DeepONet and FNO used for each example in this paper under the respective folder names. 



## ⚠️ The LOCA methodology and code cannot be used for commercial purposes (protected by a patent at the University of Pennsylvania).⚠️

