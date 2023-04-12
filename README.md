# V-Mapper - Velocity Mapper

<div style="text-align:left"><img style="width:60%; height: auto" src="https://github.com/yusuke-imoto-lab/V-Mapper/blob/main/images/VMapper_Top.jpg"/></div>

V-Mapper (velocity Mapper) is an extension of Mapper, which is a well-known topological data analysis (TDA)  method for extracting high-dimensional topological structures as a graph \[[Singh et al., 2007](https://doi.org/10.2312/SPBG/SPBG07/091-100)\].
V-Mapper is for high-dimensional data with position and velocity and describes a topological structure and flows on it simultaneously as a weighted directed graph (V-Mapper graph) by embedding given velocity data in the edges of the Mapper graph.

[Y. Imoto and Y. Hiraoka. V-Mapper: topological data analysis for high-dimensional data with velocity, 2023, Nonlinear Theory and Its Applications, IEICE](https://dx.doi.org/10.26508/lsa.202201591). 


## Installation
To install V-Mapper package, use `pip` as follows:

```
$ pip install vmapper
```

## Requirements
* Python3
* numpy
* scipy
* scikit-learn
