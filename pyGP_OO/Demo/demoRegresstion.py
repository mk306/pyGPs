#===============================================================================
#    Copyright (C) 2013
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyGPs.
# 
#    pyGPs is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyGPs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================

import pyGP_OO
from pyGP_OO.Core import *
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
# initialze input data
#-----------------------------------------------------------------
PLOT = True

#DATA
x = np.array([[2.083970427750732,  -0.821018066101379,  -0.617870699182597,  -1.183822608860694,\
              0.274087442277144,   0.599441729295593,   1.768897919204435,  -0.465645549031928,\
              0.588852784375935,  -0.832982214438054,  -0.512106527960363,   0.277883144210116,\
              -0.065870426922211,  -0.821412363806325,   0.185399443778088,  -0.858296174995998,\
               0.370786630037059,  -1.409869162416639,-0.144668412325022,-0.553299615220374]]).T
y = np.array([[4.549203746331698,   0.371985574437271,   0.711307965514790,  -0.013212893618430,   2.255473255338191,\
                  1.009915749295733,   3.744675937965029,   0.424592771793202,   1.322833652295811,   0.278298293510020,\
                  0.267229130945574,   2.200112286723833,   1.200609983308969,   0.439971697236094,   2.628580433511255,\
                  0.503774817336353,   1.942525313820564,   0.579133950013327,   0.670874423968554,   0.377353755100965]]).T
# TEST points
z = np.array([np.linspace(-1.9,1.9,101)]).T
if PLOT:
    pyGP_OO.Visual.plot.datasetPlotter(x,y,[-1.9, 1.9, -0.9, 3.9])  


#-----------------------------------------------------------------
# step 1:
# specify combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
k1 = cov.covSEiso([-1,0])
k2 = cov.covPoly([2,1,1])
k = k1*k2*6
m = mean.meanZero()
l = lik.likGauss([np.log(0.1)])
i = inf.infExact()


#-----------------------------------------------------------------
# step 2 (optional):
# specify optimization methods
#-----------------------------------------------------------------
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.max_trails = 20
#conf.min_threshold = 100
conf.covRange = [(-10,10), (-10,10), (-10,10),(-10,10),(5,6)]
conf.likRange = [(0,1)]
o = opt.Minimize(conf)
#o = opt.CG(conf)
#o = opt.BFGS(conf)
#o = opt.SCG(conf)


#-----------------------------------------------------------------
# analyze nlZ and dnlZ
#-----------------------------------------------------------------
out = gp.analyze(i,m,k,l,x,y,True)
print 'nlZ =', out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik


#-----------------------------------------------------------------
# predict without optimization
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])


#-----------------------------------------------------------------
# training (find optimal hyperparameters)
#-----------------------------------------------------------------
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print 'optimal nlZ=', nlZ_trained


#-----------------------------------------------------------------
# predict after optimization
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])


#-----------------------------------------------------------------
# More things you can do: SPARSE GP
#-----------------------------------------------------------------
# specify inducing points
n = x.shape[0]
num_u = np.fix(n/2)
u = np.linspace(-1.3,1.3,num_u).T
u  = np.reshape(u,(num_u,1))

# specify FITC covariance functions
k = cov.covSEiso([-1.0,10.0]).fitc(u)
# specify FICT inference method
i = inf.infFITC_Exact()

# The rest usage is the same as STANDARD GP
# Here we just give one example:
m = mean.meanZero()
l = lik.likGauss([np.log(0.1)])

conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.min_threshold = 20
conf.max_trails = 300
conf.covRange = [(-1,1), (-1,1)]
conf.likRange = [(0,0.2)]
o = opt.Minimize(conf)

out = gp.analyze(i,m,k,l,x,y,True)
print "[fitc] nlz=", out[0]

nlZ_trained = gp.train(i,m,k,l,x,y,o)
print '[fitc] optimal nlZ=', nlZ_trained
out = gp.predict(i,m,k,l,x,y,z)
ymF = out[0]
y2F = out[1]
mF  = out[2]
s2F = out[3]

if PLOT:
    pyGP_OO.Visual.plot.fitcPlotter(u,z,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])


#-----------------------------------------------------------------
# end of demo
#-----------------------------------------------------------------
plt.show()
print '------------------END OF DEMO----------------------'


