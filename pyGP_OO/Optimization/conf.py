import numpy as np

class random_init_conf(object):
    def __init__(self, mean, cov, lik):
        self.max_trails = None
        self.min_threshold = None
        self.mean = mean
        self.cov = cov
        self.lik = lik
        self._meanRange = [(0,1) for i in mean.hyp]
        self._covRange  = [(0,1) for i in cov.hyp]        
        self._likRange  = [(0,1) for i in lik.hyp]

    def getmr(self):
        return self._meanRange
    def setmr(self, value):
        if len(value) == len(self.mean.hyp):
            self._meanRange = value
        else:
            raise Exception('The length of meanRange is not consistent with number of mean hyparameters')
    meanRange = property(getmr,setmr)
    
    def getcr(self):
        return self._covRange
    def setcr(self, value):
        if len(value) == len(self.cov.hyp):
            self._covRange = value
        else:
            raise Exception('The length of covRange is not consistent with number of covariance hyparameters')
    covRange = property(getcr,setcr)

    def getlr(self):
        return self._likRange
    def setlr(self, value):
        if len(value) == len(self.lik.hyp):
            self._likRange = value
        else:
            raise Exception('The length of likRange is not consistent with number of liklihood hyparameters')
    likRange = property(getlr,setlr)










