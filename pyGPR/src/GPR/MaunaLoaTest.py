from random import random, randint
from GPR import gpr, kernels
from Tools import general
import numpy as np
from matplotlib import pyplot as plt

def convert_ndarray(X):
    x_t = X.flatten()
    return np.array(x_t)

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

x  = [i for (i,j) in zip(year,co2) if i < 2004]
y  = [j for (i,j) in zip(year,co2) if i < 2004]
xx = [i for (i,j) in zip(year,co2) if i >= 2004]
yy = [j for (i,j) in zip(year,co2) if i >= 2004]

x = np.array(x)
y = np.array(y)
x = x.reshape((len(x),1))
y = y.reshape((len(y),1))

covfunc = [ ['kernels.covSum'], [ ['kernels.covSEiso'],[['kernels.covProd'],[['kernels.covPeriodic'],['kernels.covSEiso']]],\
            ['kernels.covRQiso'],['kernels.covSEiso'],['kernels.covNoise'] ] ]

## SET (hyper)parameters
#logtheta = [np.log(67.*12.), np.log(66.), np.log(90.*12.), np.log(1.0), np.log(1.3), np.log(12.0), \
#            np.log(2.4), np.log(1.2*12.), np.log(0.66), np.log(0.78), np.log(1.6), np.log(0.18), np.log(0.19)]
logtheta = [np.log(67.), np.log(66.), np.log(1.3), np.log(1.0), np.log(2.4), np.log(90.), np.log(2.4), \
            np.log(1.2), np.log(0.66), np.log(0.78), np.log(1.6/12.), np.log(0.18), np.log(0.19)]
general.check_hyperparameters(covfunc,logtheta,x)

### TEST POINTS
#Xstar = np.array(range(len(X)-1,1000))
Xstar = np.arange(2004+1./24.,2024-1./24.,1./12.)
Xstar = Xstar.reshape(len(Xstar),1)
#_________________________________
# STANDARD GP:

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if False:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters
    loghyper = logtheta
    print 'initial hyperparameters: ', np.exp(loghyper)
    ### TRAINING of (hyper)parameters
    loghyper = gpr.gp_train(loghyper, covfunc, X, y)
    print 'trained hyperparameters: ',np.exp(logtheta)

## to GET prior covariance of Standard GP use:
#[Kss, Kstar] = general.feval(covfunc, logtheta, X, Xstar)    # Kss = self covariances of test cases, 
#                                                            # Kstar = cov between train and test cases
## PREDICTION
result = gpr.gp_pred(logtheta, covfunc, x, y-1.*sum(y)/len(y), Xstar) # get predictions for unlabeled data ONLY
MU = result[0] + 1.*sum(y)/len(y)
S2 = result[1]

Mu = convert_ndarray(MU)
s2 = convert_ndarray(S2)
Xs = convert_ndarray(Xstar)
sd = np.sqrt(s2)

if True:
    #plt.suptitle('logtheta:', fontsize=12)
    #plt.title(np.exp(logtheta))
    plt.plot(Xstar,Mu, 'g-', x, y, 'r-')
    plt.fill_between(Xs,Mu+2*sd,Mu-2*sd,facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.savefig('mauna.png')
    plt.show()
