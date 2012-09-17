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
Created on 31/08/2009


    This implementation (partly) follows the matlab covFunctions implementation by Rasmussen, 
    which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams.


    covariance functions/kernels to be used by Gaussian process functions. 
    Beside the graph kernels based on the regularized Laplacian 
    
        regLapKernel  - returns covariance matrix of regularized Laplacian Kernel
    
    there are two different kinds of covariance functions: simple and composite:
    
    simple covariance functions:
    
        covNoise      - independent covariance function (ie white noise)
        covSEard      - squared exponential covariance function with ard
        covSEiso      - isotropic squared exponential covariance function
    
    simple covariance matices
    
        covMatrix     - non parameterized covariance (ie kernel matrix -> no (hyper)parameters)
            
    composite covariance functions (see explanation at the bottom):
    
        covSum        - sums of (parameterized) covariance functions
        covSumMat     - sums of (parameterized) covariance functions and ONE kernel matrix
                        TODO: extend this to sum of more than one kernel matices
    
    Naming convention: all covariance functions start with "cov". A trailing
    "iso" means isotropic, "ard" means Automatic Relevance Determination, and
    "one" means that the distance measure is parameterized by a single parameter.
    
    The covariance functions are written according to a special convention where
    the exact behaviour depends on the number of input and output arguments
    passed to the function. If you want to add new covariance functions, you 
    should follow this convention if you want them to work with the function
    gpr. There are four different ways of calling
    the covariance functions:
    
    1) With no input arguments:
    
       p = covNAME
    
    The covariance function returns a string telling how many hyperparameters it
    expects, using the convention that "D" is the dimension of the input space.
    For example, calling "covSEard" returns the string 'D + 1'.
    
    2) With two input arguments:
    
       K = covNAME(logtheta, x) 
    
    The function computes and returns the covariance matrix where logtheta are
    the log og the hyperparameters and x is an n by D matrix of cases, where
    D is the dimension of the input space. The returned covariance matrix is of
    size n by n.
    
    3) With three input arguments and two output arguments:
    
       [v, B] = covNAME(loghyper, x, z)
    
    The function computes test set covariances; v is a vector of self covariances
    for the test cases in z (of length nn) and B is a (n by nn) matrix of cross
    covariances between training cases x and test cases z.
    
    4) With three input arguments and a single output:
    
        D = covNAME(logtheta, x, z)
    
    The function computes and returns the n by n matrix of partial derivatives
    of the training set covariance matrix with respect to logtheta(z), ie with
    respect to the log of hyperparameter number z.
    
    The functions may retain a local copy of the covariance matrix for computing
    derivatives, which is cleared as the last derivative is returned.
    
    About the specification of simple and composite covariance functions to be
    used by the Gaussian process function gpr:
    
       covfunc = 'kernels.covSEard'
    
    Composite covariance functions can be specified as list. For example:
    
       covfunc = ['kernels.covSum', ['kernels.covSEard','kernels.covNoise']]
    
    
    To find out how many hyperparameters this covariance function requires, we do:
    
       Tools.general.feval(covfunc)
     
    which returns the list of strings ['D + 1', 1] 
    (ie the 'covSEard' uses D+1 and 'covNoise' a single parameter).


@author: Marion Neumann (last update 08/01/10)

Substantial updates by Daniel Marthaler July 2012.
'''



import Tools
import numpy
import math

def covMatern3(loghyper=None, x=None, z=None):
    '''Matern covariance function with nu = 3/2.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * f( sqrt(3)*r ) * exp( -sqrt(d)*r )
    where f(t) = 1+t and r is the distance
    (x^p - x^q)'*inv(P)*(x^p - x^q)/2),
    where the P matrix is ell^2 times the unit matrix and
    sf2 is the signal variance.  

    The hyperparameters of the function are:
    loghyper = [ log(ell)
                log(sqrt(sf2)) ]
    '''

    if loghyper == None:                 # report number of parameters
        return 2

    sqrt3 = numpy.sqrt(3.0)

    ell = numpy.exp(loghyper[0])         # characteristic length scale
    sf2 = numpy.exp(2*loghyper[1])       # signal variance

    x = sqrt3*x/ell 
    A = numpy.sqrt(sq_dist(x))

    if z == None:       # compute covariance matix for dataset x
        A = sf2 * (1. + A ) * numpy.exp(-A)

    elif isinstance(z, int) and z == 0:  # compute derivative matrix wrt 1st parameter
        A = sf2 * numpy.exp(-A) * A * A

    elif isinstance(z, int) and z == 1:  # compute derivative matrix wrt 2nd parameter
        A = 2 * sf2 * (1. + A ) * numpy.exp(-A)

    else:               # compute covariance between data sets x and z
        z = sqrt3*z/ell
        A = sf2*numpy.ones((z.shape[0],1))         # self covariances (needed for GPR)
        K = numpy.sqrt(sq_dist(x,z))
        B = sf2 * (1. + K ) * numpy.exp(-K)         # cross covariances
        A=[A,B]
    return A

def covMatern5(loghyper=None, x=None, z=None):
    '''Matern covariance function with nu = 5/2.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * f( sqrt(3)*r ) * exp( -sqrt(d)*r )
    where f(t) = 1+t*(1+t/3) and r is the distance
    (x^p - x^q)'*inv(P)*(x^p - x^q)/2),
    where the P matrix is ell^2 times the unit matrix and
    sf2 is the signal variance.  

    The hyperparameters of the function are:
    loghyper = [ log(ell)
                log(sqrt(sf2)) ]
    '''

    if loghyper == None:                 # report number of parameters
        return 2

    sqrt5 = numpy.sqrt(5.0)
    onethird = 1./3.

    ell = numpy.exp(loghyper[0])         # characteristic length scale
    sf2 = numpy.exp(2*loghyper[1])       # signal variance

    x = sqrt5*x/ell 
    A = numpy.sqrt(sq_dist(x))

    if z == None:       # compute covariance matix for dataset x
        A = sf2 * (1. + A*(1+onethird*A) ) * numpy.exp(-A)

    elif isinstance(z, int) and z == 0:  # compute derivative matrix wrt 1st parameter
        A = sf2 * numpy.exp(-A) * A * A * onethird*(1. + A)

    elif isinstance(z, int) and z == 1:  # compute derivative matrix wrt 2nd parameter
        A = 2 * sf2 * (1. + A*(1+onethird*A) ) * numpy.exp(-A)

    else:               # compute covariance between data sets x and z
        z = sqrt5*z/ell
        A = sf2*numpy.ones((z.shape[0],1))         # self covariances (needed for GPR)
        K = numpy.sqrt(sq_dist(x,z))
        B = sf2 * (1. + K*(1.+onethird*K) ) * numpy.exp(-K)         # cross covariances
        A=[A,B]
    return A

def covSEiso(loghyper=None, x=None, z=None):

    '''Squared Exponential covariance function with isotropic distance measure.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    where the P matrix is ell^2 times the unit matrix and
    sf2 is the signal variance  

    The hyperparameters of the function are:
    loghyper = [ log(ell)
                log(sqrt(sf2)) ]
    a column vector  
    each row of x/z is a data point'''

    

    if loghyper == None:                 # report number of parameters
        return 2


#    print numpy.exp(loghyper[0])
#    print numpy.exp(loghyper[1])
    
    ell = numpy.exp(loghyper[0])         # characteristic length scale
    sf2 = numpy.exp(2*loghyper[1])       # signal variance

    x = x/ell 
    A = -sq_dist(x)/2

    if z == None:       # compute covariance matix for dataset x
        A = sf2 * numpy.exp(A)

    elif isinstance(z, int) and z == 0:  # compute derivative matrix wrt 1st parameter
        A = 2* sf2 * numpy.exp(A) * A

    elif isinstance(z, int) and z == 1:  # compute derivative matrix wrt 2nd parameter
        A = 2 * sf2 * numpy.exp(A)

    else:               # compute covariance between data sets x and z
        z = z/ell
        A = sf2*numpy.ones((z.shape[0],1))         # self covariances (needed for GPR)
        B = sf2*numpy.exp(-sq_dist(x,z)/2)         # cross covariances
        A=[A,B]
    return A

def covSEard(loghyper=None, x=None, z=None):

    '''Squared Exponential covariance function with Automatic Relevance Detemination
    (ARD) distance measure. The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space and sf2 is the signal variance. The
    hyperparameters are:
    
    loghyper = [ log(ell_1)
                  log(ell_2)
                   .
                  log(ell_D)
                  log(sqrt(sf2)) ]'''
    
    if loghyper == None:                # report number of parameters
        return 'D + 1'                  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)
    
    [n, D] = x.shape
    ell = 1/numpy.exp(loghyper[0:D])    # characteristic length scale
    sf2 = numpy.exp(2*loghyper[D])      # signal variance

    x_ = numpy.dot(numpy.diag(ell),x.transpose()).transpose()
    A = None
    if z == None:       # compute covariance matix for dataset x
        A = sf2*numpy.exp(-sq_dist(x_)/2)             
    elif isinstance(z, int) and z <D:      # compute derivative matrix wrt length scale parameters
        A = sf2*numpy.exp(-sq_dist(x_)/2) * sq_dist((numpy.array([x[:,z]])*ell[z]).transpose()) # NOTE: ell = 1/exp(loghyper) AND sq_dist is written for the transposed input!!!!

    elif isinstance(z, int) and z==D:      # compute derivative matrix wrt magnitude parameter
        A = 2*sf2*numpy.exp(-sq_dist(x_)/2)

    else:               # compute covariance between data sets x and z
        A = sf2*numpy.ones((z.shape[0],1))          # self covariances
        z = numpy.dot(numpy.diag(ell),z.transpose()).transpose()   
        B = sf2 * numpy.exp(-sq_dist(x_,z)/2)       # cross covariances
        A = [A,B]
    return A

def covPeriodic(loghyper=None, x=None, z=None):

    '''Stationary covariance function for a smooth periodic function,'
    with period p:
    k(x^p,x^q) = sf2 * exp( -2*sin^2( pi*||x^p - x^q)||/p )/ell**2 )

    The hyperparameters of the function are:
    loghyper = [ log(ell)
                log(p)
                log(sqrt(sf2)) ]
    '''

    if loghyper == None:                 # report number of parameters
        return 3

    ell = numpy.exp(loghyper[0])         # characteristic length scale
#BUG#    p   = numpy.exp(2.*loghyper[1])        # period
    p   = numpy.exp(loghyper[1])        # period
    sf2 = numpy.exp(2.*loghyper[2])       # signal variance

    A = numpy.pi*numpy.sqrt(sq_dist(x))/p

    if z == None:       # compute covariance matix for dataset x
        A = numpy.sin(A)/ell
        A = A * A
        A = sf2 *numpy.exp(-2.*A)

    elif isinstance(z, int) and z == 0:  # compute derivative matrix wrt 1st parameter
        A = numpy.sin(A)/ell
        A = A * A
        A = 4. *sf2 *numpy.exp(-2.*A) * A

    elif isinstance(z, int) and z == 1:  # compute derivative matrix wrt 2nd parameter
        R = numpy.sin(A)/ell
        A = 4 * sf2/ell * numpy.exp(-2.*R*R)*R*numpy.cos(A)*A

    elif isinstance(z, int) and z == 2:  # compute derivative matrix wrt 3rd parameter
        A = numpy.sin(A)/ell
        A = A * A
        A = 2. * sf2 * numpy.exp(-2.*A)

    else:               # compute covariance between data sets x and z
        A = sf2*numpy.ones((z.shape[0],1))         # self covariances (needed for GPR)
        B = numpy.sin(numpy.pi*numpy.sqrt(sq_dist(x,z))/p)/ell
        B = B*B
        B = sf2*numpy.exp(-2.*B)         # cross covariances
        A=[A,B]
    return A

def covRQiso(loghyper=None, x=None, z=None):

    '''Rational Quadratic covariance function with isotropic distance measure.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
    where the P matrix is ell^2 times the unit matrix,
    sf2 is the signal variance, and alpha is the shape parameter for the RQ
    covariance.  

    The hyperparameters of the function are:
    loghyper = [ log(ell)
                 log(sqrt(sf2)) 
                 log(alpha) ]
    each row of x/z is a data point'''

    if loghyper == None:                 # report number of parameters
        return 3

    ell   = numpy.exp(loghyper[0])         # characteristic length scale
    sf2   = numpy.exp(2*loghyper[1])       # signal variance
    alpha = numpy.exp(loghyper[2])         # signal variance

    x = x/ell 
    A = sq_dist(x)

    if z == None:       # compute covariance matix for dataset x
        A = sf2 * ( ( 1.0 + 0.5*A/alpha )**(-alpha) )

    elif isinstance(z, int) and z == 0:  # compute derivative matrix wrt 1st parameter
        A = sf2 * ( 1.0 + 0.5*A/alpha )**(-alpha-1) * A

    elif isinstance(z, int) and z == 1:  # compute derivative matrix wrt 2nd parameter
        A = 2.* sf2 * ( ( 1.0 + 0.5*A/alpha )**(-alpha) )

    elif isinstance(z, int) and z == 2:  # compute derivative matrix wrt 3rd parameter
        K = ( 1.0 + 0.5*A/alpha )
        A = sf2 * K**(-alpha) * (0.5*A/K - alpha*numpy.log(K) )

    else:               # compute covariance between data sets x and z
        z = z/ell
        A = sf2*numpy.ones((z.shape[0],1))         # self covariances (needed for GPR)
        B = sf2 * ( ( 1.0 + 0.5*sq_dist(x,z)/alpha )**(-alpha) )         # cross covariances
        A=[A,B]
    return A

def covRQard(loghyper=None, x=None, z=None):
    '''Rational Quadratic covariance function with Automatic Relevance Detemination
    (ARD) distance measure. The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
    
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space, sf2 is the signal variance and alpha is 
    the shape parameter for the RQ covariance. The hyperparameters are:
    
    loghyper = [ log(ell_1)
                  log(ell_2)
                   .
                  log(ell_D)
                  log(sqrt(sf2)) 
                  log(alpha)]'''
    
    if loghyper == None:                # report number of parameters
        return 'D + 2'                  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)
    
    [n, D] = x.shape
    ell = 1/numpy.exp(loghyper[0:D])    # characteristic length scale
    sf2 = numpy.exp(2*loghyper[D])      # signal variance
    alpha = numpy.exp(loghyper[D+1])

    x_ = numpy.dot(numpy.diag(ell),x.transpose()).transpose()
    A = sq_dist(x_)

    if z == None:       # compute covariance matix for dataset x
        A = sf2 * ( ( 1.0 + 0.5*A/alpha )**(-alpha) )

    elif isinstance(z, int) and z <D:      # compute derivative matrix wrt length scale parameters
        A = sf2 * ( 1.0 + 0.5*A/alpha )**(-alpha-1) * sq_dist((numpy.array([x[:,z]])/ell[z]).transpose()) # NOTE: ell = 1/exp(loghyper) AND sq_dist is written for the transposed input!!!!

    elif isinstance(z, int) and z==D:      # compute derivative matrix wrt magnitude parameter
        A = 2. * sf2 * ( ( 1.0 + 0.5*A/alpha )**(-alpha) )

    elif isinstance(z, int) and z==(D+1):      # compute derivative matrix wrt magnitude parameter
        K = ( 1.0 + 0.5*A/alpha )
        A = sf2 * K**(-alpha) * ( 0.5*A/K - alpha*numpy.log(K) )

    else:               # compute covariance between data sets x and z
        A = sf2*numpy.ones((z.shape[0],1))          # self covariances
        z = numpy.dot(numpy.diag(ell),z.transpose()).transpose()   
        B = sf2 * ( ( 1.0 + 0.5*sq_dist(x_,z)/alpha )**(-alpha) ) # cross covariances
        A = [A,B]
    return A

def covNoise(loghyper=None, x=None, z=None):
    '''Independent covariance function, ie "white noise", with specified variance.
    The covariance function is specified as:
    k(x^p,x^q) = s2 * \delta(p,q)

    where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
    which is 1 iff p=q and zero otherwise. The hyperparameter is

    loghyper = [ log(sqrt(s2)) ]'''
    if loghyper == None:                    # report number of parameters
        return 1
    
    s2 = numpy.exp(2*loghyper[0])              # noise variance
    A = None
    if z == None:       # compute covariance matix for dataset x
        A = s2*numpy.eye(x.shape[0])      
    elif isinstance(z, int):                # compute derivative matrix
        A = 2*s2*numpy.eye(x.shape[0])
    else:               # compute covariance between data sets x and z      
        A = s2*numpy.ones((z.shape[0],1))   # self covariances
        B = numpy.zeros((x.shape[0],z.shape[0]))     # zeros cross covariance by independence
        A = [A,B]
    return A   
    
def covMatrix(R_=None, Rstar_=None):
    '''This function allows for a non-paramtreised covariance.
    input:  R_:        training set covariance matrix (train by train)
            Rstar_:    cross covariances train by test
                      last row: self covariances (diagonal of test by test)
    -> no hyperparameters have to be optimised. '''
    
    if R_ == None:    # report number of parameters
        return 0
    
    A = None
    if Rstar_==None:                         # trainings set covariances
        A = R_
    elif isinstance(Rstar_, int):            # derivative matrix (not needed here!!)                             
        print 'error: NO optimization to be made in covfunc (CV is done seperatly)'
    else:                  # test set covariances  
        A = numpy.array([Rstar_[-1,]]).transpose() # self covariances for the test cases (last row) 
        B = Rstar_[0:Rstar_.shape[0]-1,:]          # cross covariances for trainings and test cases
        A = [A,B]
    return A    


def covSumMat(covfunc, loghyper=None, x=None, R=None, w=None, z=None, Rstar=None):    
    '''covSum - compose a covariance function as the sum of other covariance
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other covariance functions to do the
    actual work. 
    It's also possible to add parametrised covfunctions and non-parametrised covariance matrix.
    -> weighted sum: covfunc + w * R '''
    
    if loghyper == None:    # report number of parameters
        A = [Tools.general.feval(covfunc[0])]
        for i in range(1,len(covfunc)):
            A.append(Tools.general.feval(covfunc[i]))
        return A
    else:   # create column array
        loghyper = numpy.array([loghyper]).transpose()   
    
    [n, D] = x.shape
    
    # SET vector v (v indicates how many parameters each covfunc has)
    v = numpy.zeros((len(covfunc)+1,1))      
    v[0] = 0    # needed for technical reasons         
    for i in range(1,len(covfunc)+1):
        no_param = Tools.general.feval(covfunc[i-1])  
        if isinstance(no_param, int):
            v[i]=no_param
        else:   # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':    v[i]=int(D)
            if pram_str[1]=='+':    v[i]+=map(int, pram_str[2])    
            elif pram_str[1]=='-':  v[i]-=map(int, pram_str[2])
            else: 
                print 'error: number of parameters of '+covfunc[i] +' unknown!'     
    
    if z == None:   # compute covariance matrix
        A = numpy.zeros((n, n))             # allocate space for covariance matrix
        for i in range(1,len(covfunc)+1):   # iteration over summand functions
            f = covfunc[i-1]
            if f=='kernels.covMatrix':
                A = A + w*Tools.general.feval(f, R)  # accumulate covariances
            else:
                tmp = Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x)  # accumulate covariances
                A = A + Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x)  # accumulate covariances
           
    elif isinstance(z, int):  # compute derivative matrices   
        tmp = 0                                                                                 
        for i in range(1,len(covfunc)+1): 
            tmp += v[i]
            if z<tmp:
                j = z-v[i-1,0]; break       # j: which parameter in that covariance
                                            # i: which covariance function
        f = covfunc[i-1]
        # compute derivative
        if f=='kernels.covMatrix': 
            print 'no hyperparameters for covMatrix - nothing to be done here!'
        else:
            A = Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x, int(j))
            
    else:           # compute test set covariances
        A = numpy.zeros((z.shape[0],1))             # allocate space
        B = numpy.zeros((n,z.shape[0]))
        for i in range(1,len(covfunc)+1):
            f = covfunc[i-1]
            # compute test covariances and accumulate
            if f=='kernels.covMatrix':
                results = Tools.general.feval(f, R, Rstar)
                A = A + w*results[0]    # self covariances 
                B = B + w*results[1]    # cross covariances    
            else:
                results = Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x, z)
                A = A + results[0]    # self covariances 
                B = B + results[1]    # cross covariances        
        A = [A,B] 
    return A


def covSum(covfunc, loghyper=None, x=None, z=None):    
    '''covSum - compose a covariance function as the sum of other covariance
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other covariance functions to do the
    actual work. '''

    if loghyper == None:    # report number of parameters
        A = [Tools.general.feval(covfunc[0])]
        for i in range(1,len(covfunc)):
            A.append(Tools.general.feval(covfunc[i]))
        return A

    [n, D] = x.shape
 
    # SET vector v (v indicates how many parameters each covfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))

    v = [0]    # needed for technical reasons         
    for ii in range(1,len(covfunc)+1):
        no_param = Tools.general.feval(covfunc[ii-1])
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of covfunc is the sum
            # of all of them in this composition
            temp = 0
            for jj in xrange(len(no_param)):
                if isinstance(no_param[jj],int):
                    temp += no_param[jj]
                else: # no_param[jj] is a string
                    pram_str = no_param[jj].split(' ')
                    if pram_str[0]=='D':    temp1 = int(D)
                    if pram_str[1]=='+':    temp1 += int(pram_str[2])    
                    elif pram_str[1]=='-':  temp1 -= int(pram_str[2])
                    else: 
                        print 'error: number of parameters of '+covfunc[i] +' unknown!'
                    temp += temp1     
            v.append(temp)
        else:   # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':    temp = int(D)
            if pram_str[1]=='+':    temp += int(pram_str[2])    
            elif pram_str[1]=='-':  temp -= int(pram_str[2])
            else: 
                print 'error: number of parameters of '+covfunc[i] +' unknown!'
            v.append(temp)     

    if z == None:   # compute covariance matrix
        A = numpy.zeros((n, n))             # allocate space for covariance matrix
        for i in range(1,len(covfunc)+1):   # iteration over summand functions
            f = covfunc[i-1]
##OLD#            #A = A + Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x)  # accumulate covariances
#BUG##            A = A + Tools.general.feval(f, loghyper[v[i-1]:v[i-1]+v[i]], x)  # accumulate covariances
            A = A + Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)  # accumulate covariances
           
    elif isinstance(z, int):  # compute derivative matrices   
        tmp = 0                                                                                  
        for i in range(1,len(covfunc)+1): 
            tmp += v[i]
            if z<tmp:
                j = z-(tmp-v[i]); break       # j: which parameter in that covariance
        f = covfunc[i-1]                    # i: which covariance function
        # compute derivative
        #A = Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x, int(j))
##        A = Tools.general.feval(f, loghyper[v[i-1]:v[i-1]+v[i]], x, int(j))  
        A = Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, int(j)) 
            
    else:           # compute test set covariances
        A = numpy.zeros((z.shape[0],1))             # allocate space
        B = numpy.zeros((n,z.shape[0]))
        for i in range(1,len(covfunc)+1):
            f = covfunc[i-1] 
            # compute test covariances
            #results = Tools.general.feval(f, loghyper[int(v[i-1,0]):int(v[i-1,0])+int(v[i,0]),0], x, z)
            results = Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, z) 

            # and accumulate
            A = A + results[0]    # self covariances 
            B = B + results[1]    # cross covariances        
        A = [A,B] 
    return A


def covProd(covfunc, loghyper=None, x=None, z=None):    
    '''covProd - compose a covariance function as the product of other covariance
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other covariance functions to do the
    actual work. '''

    if loghyper == None:    # report number of parameters
        A = [Tools.general.feval(covfunc[0])]
        for i in range(1,len(covfunc)):
            A.append(Tools.general.feval(covfunc[i]))
        return A

    [n, D] = x.shape
        
    # SET vector v (v indicates how many parameters each covfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    v = [0]
    for i in range(1,len(covfunc)+1):
        no_param = Tools.general.feval(covfunc[i-1])  
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of covfunc is the sum
            # of all of them in this composition
            temp = 0
            for jj in xrange(len(no_param)):
                if isinstance(no_param[jj],int):
                    temp += no_param[jj]
                else: # no_param[jj] is a string
                    pram_str = no_param[jj].split(' ')
                    if pram_str[0]=='D':    temp1 = int(D)
                    if pram_str[1]=='+':    temp1 += int(pram_str[2])    
                    elif pram_str[1]=='-':  temp1 -= int(pram_str[2])
                    else: 
                        print 'error: number of parameters of '+covfunc[i] +' unknown!'
                    temp += temp1     
            v.append(temp)
        else:   # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':    temp = int(D)
            if pram_str[1]=='+':    temp+= int(pram_str[2])    
            elif pram_str[1]=='-':  temp-= int(pram_str[2])
            else: 
                print 'error: number of parameters of '+covfunc[i] +' unknown!'
            v.append(temp)     
          
    if z == None:   # compute covariance matrix
        A = numpy.ones((n, n))             # allocate space for covariance matrix
        for i in range(1,len(covfunc)+1):   # iteration over multiplicand functions
            f = covfunc[i-1]
            A *= Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)  # accumulate covariances
           
    elif isinstance(z, int):  # compute derivative matrices   
        tmp = 0                                                                                  
        A = numpy.ones((n, n))             # allocate space for covariance matrix
        flag = True
        for i in range(1,len(covfunc)+1): 
            tmp += v[i]
            if z<tmp and flag:
                flag = False                
                j = z-(tmp-v[i])                    # j: which parameter in that covariance
                f = covfunc[i-1]                    # i: which covariance function
                # compute derivative
                A *= Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, int(j))  
            else:
                
                f = covfunc[i-1]                    # i: which covariance function
                A *= Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)  
            
    else:           # compute test set covariances
        A = numpy.ones((z.shape[0],1))             # allocate space
        B = numpy.ones((n,z.shape[0]))
        for i in range(1,len(covfunc)+1):
            f = covfunc[i-1] 
            ## TODO: doiuble-check this!!
            # compute test covariances
            results = Tools.general.feval(f, loghyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, z)  
            # and accumulate
            A = A * results[0]    # self covariances 
            B = B * results[1]    # cross covariances        
        A = [A,B] 
    return A

def regLapKernel(R, beta, s2):
    '''Covariance/kernel matrix calculated via regluarized Laplacian.'''

    v = R.sum(axis=0)     # sum of each column
    D = numpy.diag(v)   
    
    K_R = numpy.linalg.inv(beta*(numpy.eye(R.shape[0])/s2+D-R)) # cov matrix for ALL the data
    
    ## NORMALISATION = scale to [0,1]
    ma = K_R.max(); mi = K_R.min()
    K_R = (K_R-mi)/(ma-mi)
    
    return K_R

def sq_dist(a, b=None):

    '''Compute a matrix of all pairwise squared distances
    between two sets of vectors, stored in the row of the two matrices:
    a (of size n by D) and b (of size m by D). '''

    n = a.shape[0]
    D = a.shape[1]
    m = n    

    if b == None:
        b = a.transpose()
    else:
        m = b.shape[0]
        b = b.transpose()

    C = numpy.zeros((n,m))

    for d in range(0,D):
        tt = a[:,d]
        tt = tt.reshape(n,1)
        tem = numpy.kron(numpy.ones((1,m)), tt)
        tem = tem - numpy.kron(numpy.ones((n,1)), b[d,:])
        C = C + tem * tem  

    return C
