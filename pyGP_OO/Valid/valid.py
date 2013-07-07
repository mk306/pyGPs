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

def k_fold_validation(x, y, K=10, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.

    The validation iterables are a partition of X, and each validation
    iterable is of length len(X)/K. Each training iterable is the
    complement (within X) of the validation iterable, and so each training
    iterable is of length (K-1)*len(X)/K.
    """
    # if need to shuffle data first
    if randomise:
        data = np.append(x, y, axis=1)
        np.random.shuffle(data)
        x = data[:,:-1]
        y = data[:,-1]

    n, D = x.shape
    assert n > K

    for k in xrange(K):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for i in xrange(n):
            if i % K == k:
                x_test.append(list(x[i]))
                y_test.append(list(y[i]))
            else:
                x_train.append(list(x[i]))
                y_train.append(list(y[i]))
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        yield x_train, x_test, y_train, y_test


def RMSE(predict,target):
    '''
    Root mean squared error
    '''
    error = predict - target
    return np.sqrt(np.mean(error**2))

def Prec(predict,target):
    '''
    Classification precision
    '''
    n,D = target.shape
    count = 0.
    for i in xrange(n):
        if predict[i,0] == target[i,0]:
            count += 1
    return count/n

    




