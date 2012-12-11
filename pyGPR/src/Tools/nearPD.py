import numpy as np

def nearPD(A):
    tol = 1.e-8
    # Returns the "closest" (up to tol) symmetric positive definite matrix to A.
    # Returns A if it is already Symmetric positive Definite
    flag = True
    count = 0; BOUND = 10000
    M = A.copy()
    while flag and count < BOUND:
        M = (M+M.T)/2.
        eigval, Q = np.linalg.eig(M)
        eigval = eigval.real
        Q = Q.real
        xdiag = np.diag(np.maximum(eigval, tol))
        if np.any(xdiag != np.diag(eigval)):
            pass
        else:
            flag = False
            count += 1 
        M = np.dot(Q,np.dot(xdiag,Q.T))
    if count == BOUND:
        raise Exception("This matrix caused the nearPD algorithm to not converge")
    return M

if __name__ == '__main__':
    A = np.array([[2.,-1,0,0.],[-1.,2.,-1,0],[0.,-1.,2.,-1.],[0.,0.,-1.,2.]])
    A = np.random.random((4,4))
    M = nearPD(A)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.linalg.LinAlgError:
        print "This shouldn't happen"
    print np.linalg.norm(M-A,'fro')
