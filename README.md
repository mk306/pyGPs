
==========
  pyGPs
==========

pyGPs is a library containing python implementations for Gaussian Processes (GPs).

---------------------------------------------------
NOTE: this library is currently UNDER DEVELOPMENT.
---------------------------------------------------

You can find the latest stable version (pyXGPR) HERE: http://www-kd.iai.uni-bonn.de/index.php?page=software_details&id=19.

Please feel free to contact me if you have any questions or suggestions.
Marion Neumann [marion dot neumann at iais dot fraunhofer dot de]

---------------------------------------------------
QUICK INTRODUCTION

pyGPs is a library (so far only) containing code for Gaussian Process Regression (GPR). In particular it is possible to perform relational GP (XGP) Regression. That means in addition to the attributes, it is possible to add relational knowledge among the data points in form of a (weighted) adjacancy matrix of the graph representing the relations.

pyXGPR is a python implementation of Standard and Relational GPR. Standard GPR follows the matlab implementation by Carl Edward Rasmussen and Chris Williams which is under (C) Copyright 2005 – 2007. Standard GPs are extended to utilize relational information among data points following the hidden common cause model/ XGP framework of Silva et al. Both GP variants offer methods for training and predicion. Within training the hyperparameters are optimized by maximizing the marginal likelihood by using a python implementation by Roland Memisevic of Carl Edward Rasmussen minimize.m (C) Copyright 1999 – 2006.

pyXGPR provides an implementation of Standard and Relational Gaussian Process Regression, as it is described in:

"M. Neumann, K. Kersting, Z. Xu, D. Schulz. Stacked Gaussian Process Learning. In H. Kargupta, W. Wang, editor(s), Proceedings of the 9th IEEE International Conference on Data Mining (ICDM-09), Miami, FL, USA, Dec. 6-9 2009".

Copyright (C) 2009

pyGPs is free software; you can redistribute it and/or modify  it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or  (at your option) any later version.

pyGPs is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for more details.
