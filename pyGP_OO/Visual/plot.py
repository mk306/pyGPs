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


