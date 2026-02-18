**TODO's**

Setup
[x] find solution for enabling installation without openMP
[ ] test / debug openMP acceleration -> need to troubleshoot on Intel cluster
[ ] setup CI workflow: testing on Python 3.10+ and various numpy, scipy and OpenMP versions
[ ] setup CD workflow: 1) version release on GitHub, 2) version releas on PyPI, 3) documentation generation on readthedocs

Docs
[ ] create + add example "notebooks", "scripts" and plots for interpolation / deposition walkthroughs
[ ] refactor the doc navigation with fuller index
[ ] integrate the github landing page instruction into the docs

Tests
[x] complete/add pytests

Code
[x] change integral estimation to fixed number of kernel evaluations at predefined locations:
    then, in for loop for each kernel position, e.g. 32^dim, identify the parent cell and deposit 
    that fraction into cell
    1) case 1: is kernel is much smaller than cell width -> no problem
    2) case 2: kernel spans over many cells and the number of evaluation points is low -> problem,
       since it is possible that some cells don't have an eval point falling in them 
       -> 0 weight is deposited, i.e. very noisy
[x] revisit per axis periodicity -> we do not, since cKdTree does not support that.

Paper
[ ] Figures: overview, 
[ ] Writing: introduction, methods


**Features so far** (so I don't forget)
Interpolation
+ many kernels possible
+ only global periodic boundary conditions
+ both iso- and anisotropic interpolation possible

Deposition
+ many kernels possible
+ only global periodic boundary conditions
+ non-uniform grids: varying pixel / voxel sizes for each axis possible


**Feature for future releases**
[ ] check primary selection of neighbors in anisotropic case;
    Anisotropic kNN (Marinho-pure)
	  •	Neighbor search uses Mahalanobis distance:
      \delta = |H^{-1}(r_p - r_q)|
      •	Outer neighbor satisfies \delta = 1