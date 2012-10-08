import kernels
import means
from GPR import gpr
from Tools import general
from numpy import *
from matplotlib import pyplot

STATS = True
TRAIN = True
PLOT  = False

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

## DEFINE parameterized covariance function
covfunc  = [ ['kernels.covSum'], [ ['kernels.covSEiso'],   ['kernels.covNoise'] ]]
#meanfunc = [ ['means.meanProd'], [ ['means.meanOne'],   ['means.meanLinear'] ]]      
meanfunc = [ ['means.meanZero'] ]      

## SET (hyper)parameters
covtheta = [log(1.0),log(1.1),log(1.0)]
meantheta   = []

# Build the general 'structure' for the problem
gp = {'covfunc':covfunc, 'meanfunc':meanfunc, 'covtheta':covtheta, 'meantheta':meantheta}

## CHECK (hyper)parameters and covariance function(s)
general.check_hyperparameters(gp,X) 

z = general.feval(gp['meanfunc'],gp['meantheta'], X)

### GENERATE sample observations from the GP
y = dot(linalg.cholesky(general.feval(gp['covfunc'],gp['covtheta'], X)).transpose(),random.standard_normal((l,1))) + z

### TEST POINTS
Xstar = array([linspace(-7.5,7.5,u)]).transpose() # u test points evenly distributed in the interval [-7.5, 7.5]

#_________________________________
# STANDARD GP:

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if TRAIN:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters
    gp['meantheta'] = list(random.random(len(meantheta))) 
    gp['covtheta']  = list(random.random(len(covtheta)))
    if STATS:
        print 'True hyperparameters: ', meantheta, covtheta
        print 'initial hyperparameters: ', gp['meantheta'], gp['covtheta']
        theta = gp['meantheta'] + gp['covtheta']
        print 'Initial Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Initial gradient: ', gpr.dnlml(theta ,gp, X, y)
    ### TRAINING of (hyper)parameters
    gp, val, iters = gpr.gp_train(gp, X, y)
    if STATS:
        theta = gp['meantheta'] + gp['covtheta']
        print 'trained hyperparameters in (',iters,' iterations): ', gp['meantheta'], exp(gp['covtheta'])
        print 'Log marginal likelihood after optimization = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Gradient after optimization: ', gpr.dnlml(theta ,gp, X, y)

## to GET prior covariance of Standard GP use:
[Kss, Kstar] = general.feval(gp['covfunc'], gp['covtheta'], X, Xstar)    # Kss = self covariances of test cases, 
#                                                             # Kstar = cov between train and test cases
## PREDICTION 
result = gpr.gp_pred(gp, X, y, Xstar) # get predictions for unlabeled data ONLY
MU = result[0]
S2 = result[1]

Mu = convert_ndarray(MU)
s2 = convert_ndarray(S2)
Xs = convert_ndarray(Xstar)
sd = sqrt(s2)

## plot results
if PLOT:
    pyplot.suptitle('covtheta:', fontsize=12)
    pyplot.title([gp['meantheta'],exp(gp['covtheta'])])
    pyplot.plot(Xs,Mu, 'g^-', X,y, 'ro')
    pyplot.fill_between(Xs,Mu+2.*sd,Mu-2.*sd,facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)   
    pyplot.show()

