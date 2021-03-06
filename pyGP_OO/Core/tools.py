#===============================================================================
#    Copyright (C) 2013
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyGPs.
# 
#    pyGPs is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyGPs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================


import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sqrt

def solve_chol(L, B):
    #print L.shape[0], L.shape[1], B.shape[0]
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');
    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X

def unique(x):
    # First flatten x
    y = [item for sublist in x for item in sublist]
    if isinstance(x,np.ndarray):
        n,D = x.shape
        assert(D == 1)
        y = np.array( list(set(x[:,0])) )
        y = np.reshape(y, (len(y),1))
    else:
        y = list(set(y))
    return y

def brentmin(xlow,xupp,Nitmax,tol,f,nout=None,*args):
    #def brentmin(0,smax,Nline,thr,'Psi_line',4,dalpha,alpha,hyp,K,m,likfunc,y,inffunc):
    ## BRENTMIN: Brent's minimization method in one dimension
    # code taken from
    #    Section 10.2 Parabolic Interpolation and Brent's Method in One Dimension
    #    Press, Teukolsky, Vetterling & Flannery
    #    Numerical Recipes in C, Cambridge University Press, 2002
    #
    # [xmin,fmin,funccout,varargout] = BRENTMIN(xlow,xupp,Nit,tol,f,nout,varargin)
    #    Given a function f, and given a search interval this routine isolates 
    #    the minimum of fractional precision of about tol using Brent's method.
    # 
    # INPUT
    # -----
    # xlow,xupp:  search interval such that xlow<=xmin<=xupp
    # Nitmax:     maximum number of function evaluations made by the routine
    # tol:        fractional precision 
    # f:          [y,varargout{:}] = f(x,varargin{:}) is the function
    # nout:       no. of outputs of f (in varargout) in addition to the y value
    #
    # OUTPUT
    # ------
    # fmin:      minimal function value
    # xmin:      corresponding abscissa-value
    # funccount: number of function evaluations made
    # varargout: additional outputs of f at optimum
    #
    # Copyright (c) by Hannes Nickisch 2010-01-10.

    if nout == None:
        nout = 0
    eps = sys.float_info.epsilon
    # tolerance is no smaller than machine's floating point precision
    tol = max(tol,eps)
    # Evaluate endpoints
    vargout = f(xlow,*args); fa = vargout[0][0]
    vargout = f(xupp,*args); fb = vargout[0][0]
    funccount = 2 # number of function evaluations
    # Compute the start point
    seps = sqrt(eps)
    c = 0.5*(3.0 - sqrt(5.0)) # golden ratio
    a = xlow
    b = xupp
    v = a + c*(b-a)
    w = v
    xf = v
    d = 0.
    e = 0.
    x = xf
    vargout = f(x,*args)
    fx = vargout[0][0]
    varargout = vargout[1:]
    funccount += 1
    fv = fx; fw = fx
    xm = 0.5*(a+b)
    tol1 = seps*abs(xf) + tol/3.0;
    tol2 = 2.0*tol1
    # Main loop
    while ( abs(xf-xm) > (tol2 - 0.5*(b-a)) ):
        gs = True
        # Is a parabolic fit possible
        if abs(e) > tol1:
            # Yes, so fit parabola
            gs = False
            r = (xf-w)*(fx-fv)
            q = (xf-v)*(fx-fw)
            p = (xf-v)*q-(xf-w)*r
            q = 2.0*(q-r)
            if q > 0.0:  
                p = -p
            q = abs(q)
            r = e;  e = d
            # Is the parabola acceptable
            if ( (abs(p)<abs(0.5*q*r)) and (p>q*(a-xf)) and (p<q*(b-xf)) ):
                # Yes, parabolic interpolation step
                d = p/q
                x = xf+d
                # f must not be evaluated too close to ax or bx
                if ((x-a) < tol2) or ((b-x) < tol2):
                    si = cmp(xm-xf,0)
                    if ((xm-xf) == 0): si += 1
                    d = tol1*si
            else:
                # Not acceptable, must do a golden section step
                gs = True
        if gs:
            # A golden-section step is required
            if xf >= xm: e = a-xf    
            else: 
                e = b-xf
            d = c*e
        # The function must not be evaluated too close to xf
        si = cmp(d,0)
        if (d == 0): si += 1
        x = xf + si * max(abs(d),tol1)
        vargout = f(x,*args); fu = vargout[0][0]; varargout = vargout[1:]
        funccount += 1
        # Update a, b, v, w, x, xm, tol1, tol2
        if fu <= fx:
            if x >= xf: a = xf 
            else: b = xf
            v = w; fv = fw
            w = xf; fw = fx
            xf = x; fx = fu
        else: # fu > fx
            if x < xf: 
                a = x
            else: 
                b = x 
            if ( (fu <= fw) or (w == xf) ):
                v = w; fv = fw
                w = x; fw = fu
            elif ( (fu <= fv) or ((v == xf) or (v == w)) ):
                v = x; fv = fu
        xm = 0.5*(a+b)
        tol1 = seps*abs(xf) + tol/3.0; tol2 = 2.0*tol1
        if funccount >= Nitmax:        
            # typically we should not get here
            # print 'Warning: Specified number of function evaluation reached (brentmin)'
            break
    # check that endpoints are less than the minimum found
    if ( (fa < fx) and (fa <= fb) ):
        xf = xlow; fx = fa
    elif fb < fx:
        xf = xupp; fx = fb
    fmin = fx
    xmin = xf
    vargout = [xmin,fmin,funccount]
    for vv in varargout:
        vargout.append(vv)
    return vargout

def cholupdate(R,x,sgn='+'):
    # Placeholder for a python version of MATLAB's cholupdate.  Now it is O(n^3)
    if len(x.shape) == 1:
        # Reshape x so that the dot product below is correct
        x = np.reshape(x,(x.shape[0],1))
    assert(R.shape[0] == x.shape[0])
    A = np.dot(R.T,R)
    if sgn == '+':
        R1 = A + np.dot(x,x.T)
    elif sgn == '-':
        R1 = A - np.dot(x,x.T)
    else:
        raise Exception('Sign needs to be + or - in cholupdate')
    return np.linalg.cholesky(R1).T



