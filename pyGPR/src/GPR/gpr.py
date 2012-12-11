
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
'''Created on 31/08/2009

    This implementation follows the matlab GP implementation by Rasmussen, 
    which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams. 
    It invokes minimize.py which is implemented in python by Roland Memisevic 2008, 
    following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen.

    This module provides functions for 
    - training the hyperparameters of a Standard GP (parameterised covariance function)                                 
    - prediction in the Standard GP framework
    
    (note: these also work for the XGP framework but mixture weights have to be adjusted separately!)
    
    INPUT:
    gp             a dictionary containing
        meantheta     column vector of hyperparameters of the mean function  
        meanfunc      is the mean function (string)
        meantheta     column vector of log hyperparameters (of the covariance function)  
        covfunc       is the covariance function (string)
    X              is the n by D matrix of training inputs
    y              is the column vector of targets (size n)
                   NOTE: y must have zero mean!!
    Xstar          is the nn by D matrix of test inputs
    
    R, w, Rstar    are inputs for XGP regression
    
    OUTPUT:
    gp             the structure containing the learnt hyperparameters for the given mean and cov funcs, resp.
    val            the final value of the objective function
    iters          the number of iterations used by the method
    
    
    TODOS:
    ------
    - provide a large number of random starting locations to minimize,
      and pick the best final answer.
    - box-bound the search space, so that minimize doesn't head off to
      crazily large or small values. Even better, do MAP rather than ML.
    
    

@author: Marion Neumann (last update 08/01/10)
'''

from numpy import sqrt, linalg, array, dot, zeros, size, log, diag, pi, eye, concatenate, reshape
from Tools.general import feval
from Tools.nearPD import nearPD
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg
import minimize

def gp_train(gp, X, y, R=None, w=None, CGFlag = False):
    ''' gp_train() returns the learnt hyperparameters.
    Following chapter 5.4.1 in Rasmussen and Williams: GPs for ML (2006).
    The original version (MATLAB implementation) of used optimizer minimize.m 
    is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
    The used python adaptation is by Roland Memisevic 2008.
    
    Input R and w is needed for XGP regression! '''

    # Build the parameter list that we will optimize
    theta = concatenate((gp['meantheta'],gp['covtheta']))
    if CGFlag:
        aa = cg(nlml, theta, dnlml, [gp,X,y,R,w], maxiter=100, disp=False, full_output=True)
        theta = aa[0]; fvals = aa[1]; funcCalls = aa[2]; gradcalls = aa[3]
        gvals = dnlml(theta, gp, X, y, R, w)
        if aa[4] == 1:
            print "Maximum number of iterations exceeded." 
        elif aa[4] ==  2:
            print "Gradient and/or function calls not changing."

    else:
        # Use BFGS
        aa = bfgs(nlml, theta, dnlml, [gp,X,y,R,w], maxiter=100, disp=False, full_output=True)
        theta = aa[0]; fvals = aa[1]; gvals = aa[2]; Bopt = aa[3]; funcCalls = aa[4]; gradcalls = aa[5]
        if aa[6] == 1:
            print "Maximum number of iterations exceeded." 
        elif aa[6] ==  2:
            print "Gradient and/or function calls not changing."

    mt = len(gp['meantheta'])
    gp['meantheta'] = theta[:mt]
    gp['covtheta']  = theta[mt:]

    return gp, fvals, gvals, funcCalls
    
def gp_pred(gp, X, y, Xstar, R=None, w=None, Rstar=None):
    # compute training set covariance matrix (K) and
    # (marginal) test predictions (Kss = self-cov; Kstar = corss-cov)
    if R==None:
        K = feval(gp['covfunc'], gp['covtheta'], X)                     # training covariances
        [Kss, Kstar] = feval(gp['covfunc'], gp['covtheta'], X, Xstar)   # test covariances (Kss = self covariances, Kstar = cov between train and test cases)
    else:
        K = feval(gp['covfunc'], gp['covtheta'], X, R, w)               # training covariances
        [Kss, Kstar] = feval(gp['covfunc'],gp['covtheta'], X, R, w, Xstar, Rstar)   # test covariances

    ms     = feval(gp['meanfunc'], gp['meantheta'], Xstar)
    mean_y = feval(gp['meanfunc'], gp['meantheta'], X)
    try:
        L = linalg.cholesky(K)                             # cholesky factorization of cov (lower triangular matrix)
    except linalg.linalg.LinAlgError:
        L = linalg.cholesky(nearPD(K))                 # Find the "Nearest" covariance mattrix to K and do cholesky on that
    alpha = solve_chol(L.transpose(),y-mean_y)         # compute inv(K)*(y-mean(y))
    fmu   = ms + dot(Kstar.transpose(),alpha)          # predicted means
    v = linalg.solve(L, Kstar)                  
    tmp=v*v                                     
    fs2 = Kss - array([tmp.sum(axis=0)]).transpose()  # predicted variances  
    fs2[fs2 < 0.] = 0.                                # Remove numerical noise i.e. negative variances
    return [fmu, fs2]

def solve_chol(A,B):
    return linalg.solve(A,linalg.solve(A.transpose(),B))

def nlml(theta, gp, X, y, R=None, w=None):
    n = X.shape[0]
    mt = len(gp['meantheta'])

    meantheta = theta[:mt]
    covtheta  = theta[mt:]

    # compute training set covariance matrix
    if R==None:
        K = feval(gp['covfunc'], covtheta, X)
    else:
        K = feval(gp['covfunc'], covtheta, X, R, w)     

    ms = feval(gp['meanfunc'], meantheta, X)
    
    # cholesky factorization of the covariance
    try:
        L = linalg.cholesky(K)      # lower triangular matrix
    except linalg.linalg.LinAlgError:
        L = linalg.cholesky(nearPD(K))                 # Find the "Nearest" covariance mattrix to K and do cholesky on that
    # compute inv(K)*y
    alpha = solve_chol(L.transpose(),y-ms)
    # compute the negative log marginal likelihood
    aa =  ( 0.5*dot((y-ms).transpose(),alpha) + (log(diag(L))).sum(axis=0) + 0.5*n*log(2.*pi) )
    return aa[0]
    
def dnlml(theta, gp, X, y, R=None, w=None):
    mt = len(gp['meantheta'])
    ct = len(gp['covtheta'])
    out = zeros(mt+ct)
    meantheta = theta[:mt]
    covtheta  = theta[mt:]

    W = get_W(theta, gp, X, y, R, w)

    if R==None:
        for ii in range(len(gp['meantheta'])):
            out[ii] = (W*feval(gp['meanfunc'], meantheta, X, ii)).sum()/2.
        kk = len(gp['meantheta'])
        for ii in range(len(gp['covtheta'])):
            out[ii+kk] = (W*feval(gp['covfunc'], covtheta, X, ii)).sum()/2.
    else:
        for ii in range(len(gp['meantheta'])):
            out[ii] = (W*feval(gp['meanfunc'], meantheta, X, R, w, ii)).sum()/2.
        kk = len(gp['meantheta'])
        for ii in range(len(gp['covtheta'])):
            out[ii+kk] = (W*feval(gp['covfunc'], covtheta, X, R, w, ii)).sum()/2. 
    return out
         
def get_W(theta, gp, X, y, R=None, w=None):
    '''Precompute W for convenience.'''
    n         = X.shape[0]
    mt        = len(gp['meantheta'])
    meantheta = theta[:mt]
    covtheta  = theta[mt:]
    # compute training set covariance matrix
    if R==None:
        K = feval(gp['covfunc'], covtheta, X)
    else:
        K = feval(gp['covfunc'], covtheta, X, R, w)
    ms = feval(gp['meanfunc'], meantheta, X)
    # cholesky factorization of the covariance
    try:
        L     = linalg.cholesky(K)      # lower triangular matrix
    except linalg.linalg.LinAlgError:
        L = linalg.cholesky(nearPD(K))  # Find the "Nearest" covariance mattrix to K and do cholesky on that
    alpha = solve_chol(L.transpose(),y-ms)
    W     = linalg.solve(L.transpose(),linalg.solve(L,eye(n)))-dot(alpha,alpha.transpose())
    return W
