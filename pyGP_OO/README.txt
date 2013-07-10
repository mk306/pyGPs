===============================================================================
    Copyright (C) 2013
    Marion Neumann [marion dot neumann at uni-bonn dot de]
    Daniel Marthaler [marthaler at ge dot com]
    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
 
    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
 
    This file is part of pyGPs.
 
    pyGPs is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
 
    pyGPs is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with this program; if not, see <http://www.gnu.org/licenses/>.
===============================================================================

pyGP_FN is a library containing code for Gaussian Process (GP) Regression and Classification.

pyGP_FN follows the structure and (a subset of) functionalities of the gpml matlab implementaion by Carl Edward Rasmussen and Hannes Nickisch (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21).

The current implementation has not been optimized yet and is work in progress. We appreciate any feedback.


Further it includes implementations of
- minimize.py implemented in python by Roland Memisevic 2008, following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen
- scg.py (Copyright (c) Ian T Nabney (1996-2001))
- solve_chol.py (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-09-18)
- brentmin.py (Copyright (c) by Hannes Nickisch 2010-01-10.)


installing pyGP_FN
------------------
Download the archive and extract it to any local directory.
Add the local directory to your PYTHONPATH:
	export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/

requirements
--------------
Python 2.6 or 2.7
Scipy and Numpy: two of many open-source packages for scientific computing that use the Python programming language. 


