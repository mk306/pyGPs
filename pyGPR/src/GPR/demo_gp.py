import kernels
from GPR import gpr
from Tools import general
from numpy import *
from matplotlib import pyplot

TRAIN = True
PLOT  = True

def convert_ndarray(X):
    x_t = X.flatten()
    x_p = []
    for ii in range(len(x_t)):
        x_p.append(x_t[ii])
    return array(x_p)

## GENERATE data from a noisy GP
l = 20      # number of labeled/training data 
u = 201     # number of unlabeled/test data
X = array(15*(random.random((l,1))-0.5))

## DEFINE parameterized covariance funcrion
#covfunc = [ ['kernels.covSum'], [ ['kernels.covMatern3'], ['kernels.covPeriodic'], [ ['kernels.covProd'], [['kernels.covSEard'],['kernels.covNoise']] ] ] ]
covfunc = [ ['kernels.covSum'], [ ['kernels.covPPiso'],   ['kernels.covNoise'] ]]
      
## SET (hyper)parameters
#logtheta = [log(1.0),log(1.0),log(1.1)]
logtheta = [log(1.0),log(1.1),log(2.),log(1.0)]

## CHECK (hyper)parameters and covariance function(s)
general.check_hyperparameters(covfunc,logtheta,X) 

### GENERATE sample observations from the GP
y = dot(linalg.cholesky(general.feval(covfunc, logtheta, X)).transpose(),random.standard_normal((l,1))) + 5.5

### TEST POINTS
Xstar = array([linspace(-7.5,7.5,u)]).transpose() # u test points evenly distributed in the interval [-7.5, 7.5]

#_________________________________
# STANDARD GP:

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if TRAIN:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters
    loghyper = logtheta
    print 'initial hyperparameters: ', exp(loghyper)
    ### TRAINING of (hyper)parameters
    logtheta = gpr.gp_train(loghyper, covfunc, X, y)
    print 'trained hyperparameters: ',exp(logtheta)

## to GET prior covariance of Standard GP use:
#[Kss, Kstar] = general.feval(covfunc, logtheta, X, Xstar)    # Kss = self covariances of test cases, 
#                                                             # Kstar = cov between train and test cases
## PREDICTION 
result = gpr.gp_pred(logtheta, covfunc, X, y, Xstar) # get predictions for unlabeled data ONLY
MU = result[0]
S2 = result[1]

Mu = convert_ndarray(MU)
s2 = convert_ndarray(S2)
Xs = convert_ndarray(Xstar)
sd = sqrt(s2)
## plot results
if PLOT:
    #pyplot.suptitle('logtheta:', fontsize=12)
    #pyplot.title(logtheta)
    pyplot.plot(Xs,Mu, 'g^-', X,y, 'ro')
    pyplot.fill_between(Xs,Mu+2.*sd,Mu-2.*sd,facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)   
    pyplot.show()

