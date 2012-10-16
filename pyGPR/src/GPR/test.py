import kernels
import means
from GPR import gpr, minimize
from Tools import general
from numpy import *
from matplotlib import pyplot
import scipy
from scipy.optimize import fmin_bfgs as bfgs


STATS = False

if __name__ == '__main__':

    l = 20      # number of labeled/training data 
    u = 201     # number of unlabeled/test data
    X = array(15*(random.random((l,1))-0.5))

    covfunc  = [ ['kernels.covSum'], [ ['kernels.covSEiso'],   ['kernels.covNoise'] ]]
    meanfunc = [ ['means.meanZero'] ]      

    covtheta = [log(1.0),log(1.1),log(0.1)]
    meantheta   = []

    gp = {'covfunc':covfunc, 'meanfunc':meanfunc, 'covtheta':covtheta, 'meantheta':meantheta}

    general.check_hyperparameters(gp,X) 

    z = general.feval(gp['meanfunc'],gp['meantheta'], X)
    y = dot(linalg.cholesky(general.feval(gp['covfunc'],gp['covtheta'], X)).transpose(),random.standard_normal((l,1))) + z

    if STATS:
        print 'True hyperparameters: ', exp(meantheta), exp(covtheta)
        print 'initial hyperparameters: ', gp['meantheta'], gp['covtheta']
        theta = gp['meantheta'] + gp['covtheta']
        print 'Initial Log marginal likelihood = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Initial gradient: ', gpr.dnlml(theta ,gp, X, y)

    theta = gp['meantheta'] + gp['covtheta']
    #[mintheta, fvals, iter] = minimize.run(theta, gpr.nlml, gpr.dnlml, [gp, X, y, None, None], maxnumfuneval=100)
    aa = scipy.optimize.fmin_bfgs(gpr.nlml, theta, gpr.dnlml, [gp,X,y], maxiter=100, disp=False,full_output=True)
    print "min theta = ",aa[0]
    print "fval = ",aa[1]
    print "gval = ",aa[2]
    print "Bopt = ",aa[3]
    print "funccalls ",aa[4]
    print "gradcalls ",aa[5]
    if aa[6] == 1:
        print "Maximum number of iterations exceeded."
    elif aa[6] ==  2: 
        print "Gradient and/or function calls not changing."

    #print theta
    #print gpr.nlml(theta, gp, X, y)[0]
    #print mintheta

    if STATS:
        theta = gp['meantheta'] + gp['covtheta']
        print 'trained hyperparameters in (',iters,' iterations): ', gp['meantheta'], exp(gp['covtheta'])
        print 'Log marginal likelihood after optimization = ',gpr.nlml(theta, gp, X, y)[0]
        print 'Gradient after optimization: ', gpr.dnlml(theta ,gp, X, y)

