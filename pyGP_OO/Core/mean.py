import numpy as np
import math

# mask hasn't been implemented
# the rest in original mean.py have all been implemented

class Mean(object):
    """the base function for mean function"""
    def __init__(self):
        super(Mean, self).__init__()
        self.hyp = []
        self.para = []
    def proceed(self):
        pass

    #overloading operators
    def __add__(self,mean):
        return SumOfMean(self,mean)
    def __mul__(self,other):
        # using * for both scalar and production
        # depending on the types of two objects.
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfMean(self,other)
        elif isinstance(other, Mean):
            return ProductOfMean(self,other)
        else:
            print "only numbers and Means are allowed for *"
    def __pow__(self,number):
        if isinstance(number, int) and number > 0:
            return PowerOfMean(self,number)
        else:
            print "only non-zero integers are supported for **"
    





# combinations
class ProductOfMean(Mean):
    def __init__(self,mean1,mean2):
        self.mean1 = mean1
        self.mean2 = mean2
        if mean1.hyp and mean2.hyp:
            self._hyp = mean1.hyp + mean2.hyp
        elif not mean1.hyp:
            self._hyp = mean2.hyp
        elif not mean2.hyp:
            self._hyp = mean1.hyp
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.mean1.hyp)
        self._hyp = hyp 
        self.mean1.hyp = self._hyp[:len1]
        self.mean2.hyp = self._hyp[len1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def proceed(self,x=None, der=None):
        n, D = x.shape
        A = np.ones((n, 1))                      # allocate space for mean vector
        if der == None:                          # compute mean vector
            A *= self.mean1.proceed(x)
            A *= self.mean2.proceed(x)
        elif isinstance(der, int):               # compute derivative vector  
            if der < len(self.mean1.hyp):
                A *= self.mean1.proceed(x, der)
                A *= self.mean2.proceed(x)
            elif der < len(self.hyp):
                der2 = der - len(self.mean1.hyp)
                A *= self.mean2.proceed(x, der2)
                A *= self.mean1.proceed(x) 
            else:
                raise Exception("Error: der out of range for meanProduct")            
        return A

class SumOfMean(Mean):
    def __init__(self,mean1,mean2):
        self.mean1 = mean1
        self.mean2 = mean2
        if mean1.hyp and mean2.hyp:
            self._hyp = mean1.hyp + mean2.hyp
        elif not mean1.hyp:
            self._hyp = mean2.hyp
        elif not mean2.hyp:
            self._hyp = mean1.hyp
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.mean1.hyp)
        self._hyp = hyp 
        self.mean1.hyp = self._hyp[:len1]
        self.mean2.hyp = self._hyp[len1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def proceed(self,x=None, der=None):
        n, D = x.shape
        A = np.zeros((n, 1))                      # allocate space for mean vector
        if der == None:                          # compute mean vector
            A += self.mean1.proceed(x)
            A += self.mean2.proceed(x)
        elif isinstance(der, int):               # compute derivative vector  
            if der < len(self.mean1.hyp):
                A += self.mean1.proceed(x, der)
            elif der < len(self.hyp):
                der2 = der - len(self.mean1.hyp)
                A *= self.mean2.proceed(x, der2)
            else:
                raise Exception("Error: der out of range for meanProduct")            
        return A

class ScaleOfMean(Mean):
    def __init__(self,mean,scalar):
        self.mean = mean
        if mean.hyp:
            self._hyp = [scalar] + mean.hyp 
        else:
            self._hyp = [scalar]
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        self._hyp = hyp 
        self.mean.hyp = self._hyp[1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def proceed(self,x=None,der=None):
        c = self.hyp[0]                                        # scale parameter
        if der == None:                                         # compute mean vector
            A = c * self.mean.proceed(x)                       # accumulate means
        elif isinstance(der, int) and der == 0:                 # compute derivative w.r.t. c
            A = self.mean.proceed(x)
        else:                                 
            A = c * self.mean.proceed(x,der-1) 
        return A

class PowerOfMean(Mean):
# Compose a mean function as the power of another one
#   m(x) = m0(x) ** d
    def __init__(self, mean, d):
        self.mean = mean
        if mean.hyp:
            self.hyp = [d] + mean.hyp 
        else:
            self.hyp = [d]
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        self._hyp = hyp 
        self.mean.hyp = self._hyp[1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def proceed(self,x=None,der=None):
        d = np.abs(np.floor(self.hyp[0])) 
        d = max(d,1)

        if der == None:                               # compute mean vector
            A = self.mean.proceed(x) **d              # accumulate means

        else:                                         # compute derivative vector
            A = d * self.mean.proceed(x) ** (d-1) * self.mean.proceed(x, der-1)      \
                * Tools.general.feval(f, meanhyper[1:], x, None, der-1)
    return A








# simple mean functions below
class meanZero(Mean):
    def __init__(self):
        self.hyp = []
        self.name = '0'
    def proceed(self, x=None, der=None):
        # where original meanZero function be
        n, D = x.shape
        A = np.zeros((n,1)) 
        return A

class meanOne(Mean):
    def __init__(self):
        self.hyp = []
        self.name = '1'
    def proceed(self, x=None, der=None):
        # where original meanZeo function be
        n, D = x.shape
        if der == None:                           
            A = np.ones((n,1)) 
        else:   
            A = np.zeros((n,1))
        return A

class meanConst(Mean):
    def __init__(self,hyp):
        if len(hyp) == 1:
            self.hyp = hyp
            self.name = 'c'
        else:
            print "Constant mean function is parameterized as:"
            print "m(x) = c"
            print ""
            print "The number of hyperparameters is 1"
            print "hyp = [c]"
            print "------------------------------------------------------------------"
            raise Exception("Wrong number of hyperparameters.")
    def proceed(self, x=None, der=None):
        n,D = x.shape
        if der == None:                            # evaluate mean
            A = self.hyp[0] * np.ones((n,1)) 
        elif isinstance(der, int) and der == 0:    # compute derivative vector wrt c
            A = np.ones((n,1)) 
        else:   
            A = np.zeros((n,1)) 
        return A

class meanLinear(Mean):
    def __init__(self,hyp):
        self.hyp = hyp
        self.name = 'sum_i (a_i * x_i)'
    def proceed(self, x=None, der=None):
    	n, D = x.shape
        if len(self.hyp) != D: 
            print "Linear mean function is parameterized as:"
            print "m(x) = sum_i (a_i * x_i) "
            print ""
            print "The number of hyperparameters is %d (dimension of inputs):" % D
            print "hyp = [a_1, a_2, ... , a_D]"
            print "------------------------------------------------------------------"
            raise Exception("Wrong number of hyperparameters.")
        else:
            # continue here if number of hyp is correct
            c = np.array(self.hyp)
            c = np.reshape(c,(len(c),1))
            if der == None:                             # evaluate mean
                A = np.dot(x,c)
            elif isinstance(der, int) and der < D:      # compute derivative vector wrt meanparameters
                A = np.reshape(x[:,der], (len(x[:,der]),1) ) 
            else:   
                A = np.zeros((n,1)) 
            return A



# You can make some test codes below:
    
# UPDATE
# just as the OO kernel usage: instead of hyp = np.array([1,2,3]), 
# here I use hyp = [1,2,3]

if __name__ == '__main__':
    # test1: combinations of mean functions
    m1 = meanConst([2])
    m2 = meanOne()
    myMean = ( m1*4 )
    print myMean.hyp
    myMean.hyp = [5,2]
    print m1.hyp
    print m2.hyp


    #########################################
    
    # test2: Does proceed() perform correctly compare to feval()?

    n = 20 # number of labeled/training data
    D = 1 # Dimension of input data
    x = np.array([2.083970427750732,  -0.821018066101379,  -0.617870699182597,  -1.183822608860694,\
              0.274087442277144,   0.599441729295593,   1.768897919204435,  -0.465645549031928,\
              0.588852784375935,  -0.832982214438054,  -0.512106527960363,   0.277883144210116,\
              -0.065870426922211,  -0.821412363806325,   0.185399443778088,  -0.858296174995998,\
               0.370786630037059,  -1.409869162416639,-0.144668412325022,-0.553299615220374]);
    x = np.reshape(x,(n,D))
    z = np.array([np.linspace(-1.9,1.9,101)]).T


    print myMean.proceed(x,1)

    # same result as feval
    # ALL simple mean functions work the same as feval (tested)!

    





