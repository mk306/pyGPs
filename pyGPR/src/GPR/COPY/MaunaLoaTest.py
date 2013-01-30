import kernels
import means
from gp import gp
from min_wrapper import min_wrapper
from solve_chol import solve_chol
import Tools.general
import Tools.nearPD
import numpy as np

from utils import hyperParameters, plotter, convert_to_array

if __name__ == '__main__':
    ## DATA (there were 7 samples having value -99.99 which were dropped):
    infile = '../../data/mauna.txt'
    f      = open(infile,'r')
    year   = []
    co2    = []
    for line in f:
        z  = line.split('  ')
        z1 = z[1].split('\n')
        if float(z1[0]) != -99.99:
            year.append(float(z[0]))
            co2.append(float(z1[0]))

    X  = [i for (i,j) in zip(year,co2) if i < 2004]
    y  = [j for (i,j) in zip(year,co2) if i < 2004]
    xx = [i for (i,j) in zip(year,co2) if i >= 2004]
    yy = [j for (i,j) in zip(year,co2) if i >= 2004]

    x = np.array(X)
    y = np.array(y)
    x = x.reshape((len(x),1))
    y = y.reshape((len(y),1))

    n,D = x.shape

    ## DEFINE parameterized covariance function
    covfunc = [ ['kernels.covSum'], [ ['kernels.covSEiso'],[['kernels.covProd'],[['kernels.covPeriodic'],['kernels.covSEiso']]],\
                ['kernels.covRQiso'],['kernels.covSEiso'],['kernels.covNoise'] ] ]

    ## DEFINE parameterized mean function
    meanfunc = [ ['means.meanZero'] ]      

    ## DEFINE parameterized inference and liklihood functions
    inffunc = ['inf.infExact']
    likfunc = ['lik.likGauss']

    ## SET (hyper)parameters
    hyp = hyperParameters()

    ## SET (hyper)parameters for covariance and mean
    hyp.cov = np.array([np.log(67.), np.log(66.), np.log(1.3), np.log(1.0), np.log(2.4), np.log(90.), np.log(2.4), \
                np.log(1.2), np.log(0.66), np.log(0.78), np.log(1.6/12.), np.log(0.18), np.log(0.19)])
    hyp.mean = np.array([])

    sn = 0.1
    hyp.lik = np.array([np.log(sn)])

    #_________________________________
    # STANDARD GP:
    ### TEST POINTS
    xs = np.arange(2004+1./24.,2024-1./24.,1./12.)
    xs = xs.reshape(len(xs),1)

    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]
    plotter(xs,ym,ys2,x,y)
    
    #vargout = min_wrapper(hyp,gp,'CG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    #hyp = vargout[0]
    #vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    #ym = vargout[0]; ys2 = vargout[1]
    #m  = vargout[2]; s2  = vargout[3]
    #plotter(xs,ym,ys2,x,y)
    
