from GPR import gpr, kernels, means
from Tools import general, nearPD
import numpy
from numpy import array, random, sqrt, log, exp, dot, linalg, linspace, concatenate, tile, ravel, meshgrid, reshape, fix
from random import sample
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as plt3

STATS      = True
TRAIN_BFGS = True
TRAIN_CG   = False
PLOT       = True

def convert_ndarray(X):
    x_t = X.flatten()
    x_p = []
    for ii in range(len(x_t)):
        x_p.append(x_t[ii])
    return array(x_p)

## GENERATE data from a noisy GP
lX = 201   # number of labeled/training data 
nX = 201    # number of unlabeled/test data
D = 1       # Dimension of input data
X = array(15*(random.random((lX,D))-0.5))

# GENERATE Inducing variables (take a random subset of the data)
#nu = sample( range(lX),int(fix(lX/8)) ) 
nu = sorted(sample( range(lX),int(fix(lX/8)) ) )
u  = X[nu,:]

## DEFINE parameterized covariance function
covfunc = [ ['kernels.covSum'], [ ['kernels.covSEiso'],['kernels.covNoise'] ] ]
covfuncF = [ ['kernels.covFITC'], covfunc, u]

#meanfunc = [ ['means.meanProd'], [ ['means.meanOne'], ['means.meanLinear'] ] ]       
meanfunc = [ ['means.meanZero'] ]      

## SET (hyper)parameters
covtheta = array([log(1.1),log(1.2),log(.5)])

#meantheta   = array([log(2.0),log(2.0)])
meantheta   = array([])

# Build the general 'structure' for the problem
gp = {'covfunc':covfuncF, 'meanfunc':meanfunc, 'covtheta':covtheta, 'meantheta':meantheta}

## CHECK (hyper)parameters and covariance function(s)
general.check_hyperparameters(gp,X) 

z = general.feval(gp['meanfunc'],gp['meantheta'], X)
A = general.feval(covfunc,gp['covtheta'], X)

### GENERATE sample observations from the GP
try:
    y = dot(linalg.cholesky(A).transpose(),random.standard_normal((lX,1))) + z
except linalg.linalg.LinAlgError:
    y = dot(linalg.cholesky(nearPD.nearPD(A)).transpose(),random.standard_normal((lX,1))) + z
#_________________________________
# STANDARD GP:
# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if TRAIN_BFGS:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters
    gp['meantheta'] = meantheta + .1*random.random(len(meantheta)) 
    gp['covtheta']  = covtheta  + .1*random.random(len(covtheta))
    if STATS:
        print 'True hyperparameters: ', meantheta, covtheta
        print 'initial hyperparameters: ', gp['meantheta'], gp['covtheta']
        theta = concatenate((gp['meantheta'],gp['covtheta']))
        print 'Initial Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Initial gradient: ', gpr.dnlml(theta ,gp, X, y)
    ### TRAINING of (hyper)parameters
    print "BFGS: "
    gp, fvals, gvals, funcCalls = gpr.gp_train(gp, X, y, CGFlag=False)
    if STATS:
        theta = concatenate((gp['meantheta'],gp['covtheta']))
        print 'trained hyperparameters in (',funcCalls,' function calls): ', gp['meantheta'], exp(gp['covtheta'])
        print 'Log marginal likelihood after optimization = ', fvals
        print 'Gradient after BFGS optimization: ', gvals

if TRAIN_CG:
    print "CG Iteration:"
    gp['meantheta'] = meantheta + .01*random.random(len(meantheta)) 
    gp['covtheta']  = covtheta  + .01*random.random(len(covtheta))
    if STATS:
        print 'initial hyperparameters: ', gp['meantheta'], gp['covtheta']
        theta = concatenate( (gp['meantheta'],gp['covtheta']) )
        print 'Initial Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Initial gradient: ', gpr.dnlml(theta ,gp, X, y)
    gp, fvals, gvals, funcCalls = gpr.gp_train(gp, X, y, CGFlag=True)
    if STATS:
        theta = concatenate((gp['meantheta'],gp['covtheta']))
        print 'trained hyperparameters in (',funcCalls,' function calls): ', gp['meantheta'], exp(gp['covtheta'])
        print 'Log marginal likelihood after optimization = ', fvals
        print 'Gradient after CG optimization: ', gvals

## PREDICTION 
Xstar = array([linspace(-7.5,7.5,nX)]).transpose()
result = gpr.gp_pred(gp, X, y, Xstar) # get predictions for unlabeled data ONLY
MU = result[0]
S2 = result[1]

Mu = convert_ndarray(MU)
s2 = convert_ndarray(S2)
Xs = convert_ndarray(Xstar)
sd = sqrt(s2)

## Plot results
if PLOT:
    plt.suptitle('Hyperparameters: ', fontsize=10)
    plt.title([gp['meantheta'],exp(gp['covtheta'])])
    plt.plot(Xs,Mu, 'g^-', X,y, 'ro')
    plt.plot(u,y[nu], 'bo')
    plt.fill_between(Xs,Mu+2.*sd,Mu-2.*sd,facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)   
    #plt.savefig('FITC.png')
    plt.show()
