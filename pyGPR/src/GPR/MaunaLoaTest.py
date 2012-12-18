from random import random, randint
from GPR import gpr, kernels, means
from Tools import general
import numpy as np
from matplotlib import pyplot as plt

def convert_ndarray(X):
    x_t = X.flatten()
    return np.array(x_t)

TRAIN   = True
PLOT    = True

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

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((len(X),1))
    y = y.reshape((len(y),1))

    ## DEFINE parameterized covariance function
    covfunc = [ ['kernels.covSum'], [ ['kernels.covSEiso'],[['kernels.covProd'],[['kernels.covPeriodic'],['kernels.covSEiso']]],\
                ['kernels.covRQiso'],['kernels.covSEiso'],['kernels.covNoise'] ] ]

    ## DEFINE parameterized mean function
    meanfunc = [ ['means.meanZero'] ]      

    ## SET (hyper)parameters for covariance and mean
    #covtheta = [np.log(67.), np.log(66.), np.log(1.3), np.log(1.0), np.log(2.4), np.log(90.), np.log(2.4), \
    #            np.log(1.2), np.log(0.66), np.log(0.78), np.log(1.6/12.), np.log(0.18), np.log(0.19)]
    covtheta = [5.03072168e+00, 6.08142314e+00, 4.39246925e-01,-3.09533121e-04,5.66245799e-01, 5.51188858e+00, 5.66245799e-01, 1.14557670e+00,\
                6.33121196e-01,-3.09109716e+00,-2.03797279e+00,-1.72264753e+00,-1.65551913e+00]
    meantheta   = np.array([])

    # Build the general 'structure' for the problem
    gp = {'covfunc':covfunc, 'meanfunc':meanfunc, 'covtheta':covtheta, 'meantheta':meantheta}

    ## CHECK (hyper)parameters and covariance function(s)
    general.check_hyperparameters(gp,X) 

    ### TEST POINTS
    Xstar = np.arange(2004+1./24.,2024-1./24.,1./12.)
    Xstar = Xstar.reshape(len(Xstar),1)

    ### TRAINING GP
    if TRAIN:
        print 'Rassmussen  hyperparameters: ', meantheta, covtheta
        theta = np.concatenate((gp['meantheta'],gp['covtheta']))
        print 'Initial Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Initial gradient: ', gpr.dnlml(theta ,gp, X, y)
        gp, fvals, gvals, funcCalls = gpr.gp_train(gp, X, y, CGFlag=False)
        print 'Trained  hyperparameters: ', gp['meantheta'], gp['covtheta']
        theta = np.concatenate((gp['meantheta'],gp['covtheta']))
        print 'Trained Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Trained gradient: ', gpr.dnlml(theta ,gp, X, y)

    #Rasmussen  hyperparameters:  [] [4.2046926193909657, 4.1896547420264252, 0.26236426446749106, 0.0, 0.87546873735389985, 4.499809670330265, 0.87546873735389985, 0.18232155679395459, \
    #                                 -0.41551544396166579, -0.24846135929849961, -2.0149030205422647, -1.7147984280919266, -1.6607312068216509]
    #Initial Log marginal likelihood =  143.216881154
    #Initial gradient:  [  5.06627083e+00  -4.86334720e+01  -3.37480332e+01   7.07427593e+02
    #                      1.58567370e+01  -8.30932355e+00   1.58567370e+01   3.88141869e+00
    #                      -6.61672539e+00  -3.53197787e-02   1.57458809e+00   9.01825496e-01
    #                      -4.43445087e+00]
    #Maximum number of iterations exceeded.
    #Trained  hyperparameters:  [] [  5.03072168e+00,   6.08142314e+00,   4.39246925e-01,  -3.09533121e-04,\
    #                                 5.66245799e-01,   5.51188858e+00,   5.66245799e-01,   1.14557670e+00,\
    #                                 6.33121196e-01,  -3.09109716e+00,  -2.03797279e+00,  -1.72264753e+00,\
    #                                 -1.65551913e+00]
    #Trained Log marginal likelihood =  109.230355706
    #Trained gradient:  [ -3.17178548e-07  -6.23054802e-07  -3.57463780e-06   2.76595120e-03
    #                     -8.27656933e-07   2.67267118e-06  -8.27656933e-07  -5.04817520e-06
    #                     -1.49336394e-07  -2.39738572e-06   1.83167783e-05  -1.48770989e-05
    #                     -1.30199073e-05]

    result = gpr.gp_pred(gp, X, y, Xstar) 
    MU = result[0] 
    S2 = result[1]
    Mu = convert_ndarray(MU)
    s2 = convert_ndarray(S2)
    Xs = convert_ndarray(Xstar)
    sd = np.sqrt(s2)

    if PLOT:
        plt.plot(Xstar,Mu, 'g-', X, y, 'r.-')
        plt.fill_between(Xs,Mu+2.*sd,Mu-2.*sd,facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
        #plt.savefig('mauna.png')
        plt.show()
    
