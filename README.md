# V-Mapper - Velocity Mapper

A topological data analysis (TDA) method for high-dimensional data, Mapper, represents a topology of data as a simplicial complex or graph abstraction based on the nerve of clusters \[[Singh et al., 2007](https://doi.org/10.2312/SPBG/SPBG07/091-100)\]. V-Mapper (velocity Mapper) is an extension of Mapper for high-dimensional data with position and velocity. 
V-Mapper describes a topological structure and flows on it simultaneously as a weighted directed graph (V-Mapper graph) by embedding given velocity data in the edges of the Mapper graph.

<div style="text-align:left"><img style="width:60%; height: auto" src="https://github.com/yusuke-imoto-lab/V-Mapper/blob/main/images/VMapper_Top.jpg"/></div>

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
