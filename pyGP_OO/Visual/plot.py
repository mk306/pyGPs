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

import numpy as np
import matplotlib.pyplot as plt


def datasetPlotter(x,y, axisvals):
    plt.figure()
    plt.plot(x,y,'b+',markersize=12)
    plt.axis(axisvals)
    plt.grid()
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.draw()

def standardPlotter(xs,ym,ys2,x,y,axisvals=None,file=None):
    plt.figure()
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.grid()
    if axisvals:
        plt.axis(axisvals)
    plt.xlabel('input x')
    plt.ylabel('output y')
    if file and isinstance(file,str):
        plt.savefig(file)
    plt.draw()

def fitcPlotter(u,xs,ym,ys2,x,y,axisvals=None,file=None):
    plt.figure()
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.grid()
    if axisvals:
        plt.axis(axisvals)
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.plot(u,np.ones_like(u),'kx',markersize=12)
    if file and isinstance(file,str):
        plt.savefig(file)
    plt.draw()
    



