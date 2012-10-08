import Tools
import numpy
import math

def meanConst(meanhyper=None, x=None, z=None):
    ''' 
    Constant mean function. The mean function is parameterized as:
      m(x) = c
    
    The hyperparameter is:
      meanhyper = [ c ]
    '''
    if meanhyper == None:                     # report number of parameters
        return 1

    n,D = x.shape
    c = meanhyper;

    if z == None:                            # evaluate mean
        A = c*numpy.ones((n,1)) 
    elif isinstance(z, int) and z == 0:      # compute derivative vector wrt c
        A = numpy.ones((n,1)) 
    else:   
        A = numpy.zeros((n,1)) 
    return A

def meanLinear(meanhyper=None, x=None, z=None):
    ''' 
    Linear mean function. The mean function is parameterized as:
      m(x) = sum_i ai * x_i 
    
    The hyperparameter is:
      meanhyper = [ a_1 
                    a_2
                    ...
                    a_D ]
    '''

    if meanhyper == None:                     # report number of parameters
        return 'D + 0'
    n, D = x.shape

    c = numpy.array(meanhyper)
    c = numpy.reshape(c,(len(c),1))
    if z == None:                            # evaluate mean
        A = numpy.dot(x,c.T)
    elif isinstance(z, int) and z < D:      # compute derivative vector wrt meanparameters
        A = x[:,z] 
    else:   
        A = numpy.zeros((n,1)) 
    return A

def meanOne(meanhyper=None, x=None, z=None):
    ''' 
    One mean function. The mean function does not have any parameters
      m(x) = 1
    
    '''

    if meanhyper == None:                     # report number of parameters
        return 0

    n,D = x.shape

    if z == None:                            # evaluate mean
        A = numpy.ones((n,1)) 
    else:   
        A = numpy.zeros((n,1)) 
    return A

def meanZero(meanhyper=None, x=None, z=None):
    ''' 
    Zero mean function. The mean function does not have any parameters
      m(x) = 1
    
    '''

    if meanhyper == None:                     # report number of parameters
        return 0

    n, D = x.shape

    A = numpy.zeros((n,1)) 
    return A

def meanProd(meanfunc, meanhyper=None, x=None, z=None):
    ''' meanProd - compose a mean function as the product of other mean
     functions. This function doesn't actually compute very much on its own, it
     merely does some bookkeeping, and calls other mean functions to do the
     actual work. 
    '''

    if meanhyper == None:    # report number of parameters
        A = [Tools.general.feval(meanfunc[0])]
        for i in range(1,len(meanfunc)):
            A.append(Tools.general.feval(meanfunc[i]))
        return A

    [n, D] = x.shape

    # SET vector v (v indicates how many parameters each meanfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    v = [0]
    for i in range(1,len(meanfunc)+1):
        no_param = Tools.general.feval(meanfunc[i-1])
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of meanfunc is the sum
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
                        print 'error: number of parameters of '+meanfunc[i] +' unknown!'
                    temp += temp1
            v.append(temp)
        else:   # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':    temp = int(D)
            if pram_str[1]=='+':    temp+= int(pram_str[2])
            elif pram_str[1]=='-':  temp-= int(pram_str[2])
            else:
                print 'error: number of parameters of '+meanfunc[i] +' unknown!'
            v.append(temp)

    if z == None:                          # compute mean vector
        A = numpy.ones((n, 1))             # allocate space for mean vector
        for i in range(1,len(meanfunc)+1): # iteration over multiplicand functions
            f = meanfunc[i-1]
            A *= Tools.general.feval(f,meanhyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)  # accumulate meanss

    elif isinstance(z, int):               # compute derivative vector   
        tmp = 0
        A = numpy.ones((n, 1))             # allocate space for derivative vector
        flag = True
        for i in range(1,len(meanfunc)+1):
            tmp += v[i]
            if z<tmp and flag:
                flag = False
                f = meanfunc[i-1]                   # i: which mean function
                j = z-(tmp-v[i])                    # j: which parameter in that mean
                # compute derivative
                A *= Tools.general.feval(f, meanhyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, int(j))
            else:
                f = meanfunc[i-1]                    # i: which mean function
                A *= Tools.general.feval(f, meanhyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)

    else:                            
        A = numpy.zeros((n,1))
    return A

def meanSum(meanfunc, meanhyper=None, x=None, z=None):
    '''covSum - compose a mean function as the sum of other mean
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other mean functions to do the
    actual work. '''

    if meanhyper == None:    # report number of parameters
        A = [Tools.general.feval(meanfunc[0])]
        for i in range(1,len(meanfunc)):
            A.append(Tools.general.feval(meanfunc[i]))
        return A

    [n, D] = x.shape

    # SET vector v (v indicates how many parameters each meanfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))

    v = [0]    # needed for technical reasons         
    for ii in range(1,len(meanfunc)+1):
        no_param = Tools.general.feval(meanfunc[ii-1])
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of meanfunc is the sum
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
                        print 'error: number of parameters of '+meanfunc[i] +' unknown!'
                    temp += temp1
            v.append(temp)
        else:   # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':    temp = int(D)
            if pram_str[1]=='+':    temp += int(pram_str[2])
            elif pram_str[1]=='-':  temp -= int(pram_str[2])
            else:
                print 'error: number of parameters of '+meanfunc[i] +' unknown!'
            v.append(temp)

    if z == None:                           # compute mean vector
        A = numpy.zeros((n, 1))             # allocate space for mean vector
        for i in range(1,len(meanfunc)+1):   # iteration over summand functions
            f = meanfunc[i-1]
            A = A + Tools.general.feval(f, meanhyper[sum(v[0:i]):sum(v[0:i])+v[i]], x)  # accumulate means

    elif isinstance(z, int):                # compute derivative vector
        tmp = 0
        for i in range(1,len(meanfunc)+1):
            tmp += v[i]
            if z<tmp:
                j = z-(tmp-v[i]); break     # j: which parameter in that mean
        f = meanfunc[i-1]                    # i: which mean function
        # compute derivative
        A = Tools.general.feval(f, meanhyper[sum(v[0:i]):sum(v[0:i])+v[i]], x, int(j))

    else:                                   # compute test set means
        A = numpy.zeros((n,1))
    return A

def meanScale(meanfunc, meanhyper=None, x=None, z=None):
    '''Compose a mean function as a scaled version of another one
    k(x^p,x^q) = sf2 * k0(x^p,x^q)
    
    The hyperparameter is :
    
    meanhyper = [ log(sf2) ]

    This function doesn't actually compute very much on its own. it merely does
    some bookkeeping, and calls another mean function to do the actual work.
    '''

    if meanhyper == None:    # report number of parameters
        A = [1]
        A.append( Tools.general.feval(meanfunc[0]) )
        return A

    c = meanhyper[0]         # scale parameter

    if z == None:            # compute mean vector
        f = meanfunc[0]
        A = c * Tools.general.feval(f, meanhyper[1:], x)  # accumulate means

    elif isinstance(z, int) and z == 0:                # compute derivative w.r.t. c
        f = meanfunc[0]
        A = Tools.general.feval(f, meanhyper[1:], x)

    else:                                   
        f = meanfunc[0]
        A = c * Tools.general.feval(f, meanhyper[1:], x, z-1)
    return A

def meanPow(meanfunc, meanhyper=None, x=None, z=None):
    '''Compose a mean function as the power of another one
      m(x) = m0(x) ** d
    
    This function doesn't actually compute very much on its own. it merely does
    some bookkeeping, and calls another mean function to do the actual work.
    '''

    if meanhyper == None:    # report number of parameters
        A = [1]
        A.append( Tools.general.feval(meanfunc[0]) )
        return A

    d = meanhyper[0]         # degree
    if numpy.abs(d-numpy.round(d)) < 1e-8:     # remove numerical error from format of parameter
        d = int(round(d))
    assert(d == int(d) and d > 0)           # Only allowed degrees > 0
    d = int(d)

    if z == None:            # compute mean vector
        f = meanfunc[0]
        A = ( Tools.general.feval(f, meanhyper[1:], x) )** d  # accumulate means

    else:                # compute derivative vector
        f = meanfunc[0]
        A = d * (Tools.general.feval(f, meanhyper[1:], x))**(d-1) \
                * Tools.general.feval(f, meanhyper[1:], x, z-1)
    return A
