#===============================================================================
#    Copyright (C) 2009  
#    Marion Neumann [marion dot neumann at iais dot fraunhofer dot de]
#    Zhao Xu [zhao dot xu at iais dot fraunhofer dot de]
#    Supervisor: Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyXGPR.
# 
#    pyXGPR is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyXGPR is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License 
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================
'''
Created on 15.09.2009

Demo_GP: Standard GP to demonstrate prediction (and training) for generated noisy data.

@author: Marion Neumann

Substantial updates by Daniel Marthaler July 2012.
'''
import kernels
from GPR import gpr
from Tools import general
from numpy import *
from matplotlib import pyplot

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def convert_ndarray(X):
    x_t = X.flatten()
    x_p = []
    for ii in range(len(x_t)):
        x_p.append(x_t[ii])
    return array(x_p)

## DATA:
## necessary data for   GPR  X, y, Xstar     
## NOTE: y must have zero mean!!
"""Update: D. Marthaler added nonzero (deterministic) mean on 26-07-2012"""

## GENERATE data from a noisy GP
l = 20      # number of labeled/training data 
u = 201     # number of unlabeled/test data
X = array(15*(random.random((l,1))-0.5))

## DEFINE parameterized covariance funcrion
covfunc = [ ['kernels.covSum'], [ ['kernels.covMatern3'], ['kernels.covPeriodic'], [ ['kernels.covProd'], [['kernels.covSEard'],['kernels.covNoise']] ] ] ]
covfunc = [ ['kernels.covSum'], [ ['kernels.covSEard'],   ['kernels.covNoise'] ]]
#covfunc = [ ['kernels.covSum'], [ ['kernels.covMatern3'], ['kernels.covRQard']]]
#covfunc = [['kernels.covSEiso']]
#covfunc = [['kernels.covMatern3']]
#covfunc = [ ['kernels.covSEard'] ]

#print '---'
#print general.feval(covfunc)
#print flatten(general.feval(covfunc))
#print '---'
      
## SET (hyper)parameters
logtheta = [log(1.0), log(1.1), log(1.0)]#, log(1.1), log(1.2)]#, log(1.0), log(1.1), log(1.0)]

## CHECK (hyper)parameters and covariance function(s)
general.check_hyperparameters(covfunc,logtheta,X)
### # MOVED to Tools.check_hyperparameters (01/08/2012)
###try:
###    assert( sum(flatten(general.feval(covfunc))) - len(logtheta) == 0 )
###except TypeError:
###    try:    
###        assert(general.get_nb_param(flatten(general.feval(covfunc)), X.shape) - len(logtheta) == 0)
###    except AssertionError:
###        print 'ERROR: number of hyperparameters does not match given covariance function:', general.get_nb_param(flatten(general.feval(covfunc)), X.shape), 'hyperparameters needed (', len(logtheta), 'given )!'
###        exit()
###except AssertionError:
###    print 'ERROR: number of hyperparameters does not match given covariance function:', sum(flatten(general.feval(covfunc))), 'hyperparameters needed (', len(logtheta), 'given )!'
###    exit()
    
#print 'hyperparameters: ', exp(logtheta)

### GENERATE sample observations from the GP
y = dot(linalg.cholesky(general.feval(covfunc, logtheta, X)).transpose(),random.standard_normal((l,1))) + 5.5

### TEST POINTS
Xstar = array([linspace(-7.5,7.5,u)]).transpose() # u test points evenly distributed in the interval [-7.5, 7.5]

#_________________________________
# STANDARD GP:

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if True:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters by -1
    loghyper = logtheta
    print 'initial hyperparameters: ', exp(loghyper)
    ### TRAINING of (hyper)parameters
    logtheta = gpr.gp_train(loghyper, covfunc, X, y)
    print 'trained hyperparameters: ',exp(logtheta)

## to GET prior covariance of Standard GP use:
[Kss, Kstar] = general.feval(covfunc, logtheta, X, Xstar)    # Kss = self covariances of test cases, 
#                                                             # Kstar = cov between train and test cases
## PREDICTION 
result = gpr.gp_pred(logtheta, covfunc, X, y, Xstar) # get predictions for unlabeled data ONLY
MU = result[0]
S2 = result[1]


Mu = convert_ndarray(MU)
s2 = convert_ndarray(S2)
Xs = convert_ndarray(Xstar)


## plot results
if True:
    pyplot.suptitle('logtheta:', fontsize=12)
    pyplot.title(logtheta)
    pyplot.plot(Xs,Mu, 'g^-', X,y, 'ro')
    pyplot.fill_between(Xs,Mu+s2,Mu-s2,facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    pyplot.fill_between(Xs,Mu+2.*s2,Mu-2.*s2,facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)
    pyplot.fill_between(Xs,Mu+3.*s2,Mu-3.*s2,facecolor=[0.,1.0,0.0,0.2],linewidths=0.0)
   
    pyplot.show()
