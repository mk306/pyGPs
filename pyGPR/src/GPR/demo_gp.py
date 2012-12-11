import kernels
import means
from GPR import gpr
from Tools import general
from numpy import array, random, sqrt, log, exp, dot, linalg, linspace, concatenate, tile, ravel, meshgrid, reshape
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
l = 20      # number of labeled/training data 
u = 201     # number of unlabeled/test data
D = 2       # Dimension of input data
X = array(15*(random.random((l,D))-0.5))

## DEFINE parameterized covariance function
covfunc = [ ['kernels.covProd'], [ [['kernels.covMask'], [[0], ['kernels.covMatern']]],['kernels.covMatern']]]

#meanfunc = [ ['means.meanProd'], [ ['means.meanOne'], ['means.meanLinear'] ] ]       
meanfunc = [ ['means.meanZero'] ]      

## SET (hyper)parameters
covtheta = array([log(1.0),log(1.1),log(3.0),log(1.0),log(1.1),log(5.0)])

#meantheta   = array([log(2.0),log(2.0)])
meantheta   = array([])

# Build the general 'structure' for the problem
gp = {'covfunc':covfunc, 'meanfunc':meanfunc, 'covtheta':covtheta, 'meantheta':meantheta}

## CHECK (hyper)parameters and covariance function(s)
general.check_hyperparameters(gp,X) 

z = general.feval(gp['meanfunc'],gp['meantheta'], X)

### GENERATE sample observations from the GP
y = dot(linalg.cholesky(general.feval(gp['covfunc'],gp['covtheta'], X)).transpose(),random.standard_normal((l,1))) + z

#_________________________________
# STANDARD GP:

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS***
### TRAINING GP
if TRAIN_BFGS:
    print 'GP: ...training'
    ### INITIALIZE (hyper)parameters
    gp['meantheta'] = meantheta + .01*random.random(len(meantheta)) 
    gp['covtheta']  = covtheta + .01*random.random(len(covtheta))
    gp['covtheta'][2]  = covtheta[2]
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
    gp['covtheta']  = covtheta + .01*random.random(len(covtheta))
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
## Plot results
if PLOT:
    # Note, we only need test points if we are going to plot the result.
    ### TEST POINTS
    Flag = True
    if D == 1:
        Xstar = array([linspace(-7.5,7.5,u)]).transpose() # u test points evenly distributed in the interval [-7.5, 7.5]
    elif D == 2:
        v = linspace(-7.5,7.5,u) # u test points evenly distributed in the interval [-7.5, 7.5]
        v = reshape( v, (len(v),1) )
        Z1,Z2 = meshgrid(v,v)
        aa = zip(ravel(Z1),ravel(Z2))
        Xstar = reshape(aa,(len(aa),2))
    else:
        print "Too many dimensions to plot!"
        Flag = False
    # to GET prior covariance of Standard GP use:
    #[Kss, Kstar] = general.feval(gp['covfunc'], gp['covtheta'], X, Xstar)    # Kss = self covariances of test cases, Kstar = cov between train and test cases
    if Flag:
        result = gpr.gp_pred(gp, X, y, Xstar) # get predictions for unlabeled data ONLY
        MU = result[0]
        S2 = result[1]

        Mu = convert_ndarray(MU)
        s2 = convert_ndarray(S2)
        Xs = convert_ndarray(Xstar)
        sd = sqrt(s2)

        if D == 1:
            plt.suptitle('Hyperparameters: ', fontsize=10)
            plt.title([gp['meantheta'],exp(gp['covtheta'])])
            plt.plot(Xs,Mu, 'g^-', X,y, 'ro')
            plt.fill_between(Xs,Mu+2.*sd,Mu-2.*sd,facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)   
            plt.show()
        elif D == 2:
            fig = plt.figure()
            ax  = plt3.Axes3D(fig)
            plt.suptitle('hyperparameters: ', fontsize=10)
            plt.title([gp['meantheta'],exp(gp['covtheta'])])
            ax.scatter3D(ravel(X[:,0]),ravel(X[:,1]),ravel(y),'o',c = 'r')
            Mu = reshape( Mu,(u,u) )
            sd = reshape( sd,(u,u) )
            ax.plot_surface(Z1,Z2,Mu, color='g', alpha = 0.25)
            ax.plot_surface(Z1,Z2,Mu+2.*sd, color='g', alpha = 0.25)
            ax.plot_surface(Z1,Z2,Mu-2.*sd, color='g', alpha = 0.25)
            plt.show()
        else:
            print "Cannot plot in more than 3 dimensions!"

