import numpy as np
import scipy

class Step(object):
    '''
    class for calculating the next increment in parameters
    for M-estimation of nonlinear models
    Notes
    -----

    **Attributes**

    n : number of rows of errormatrix (no of observations)

    p : number of columns of errormatrix 
        (no of columns in endogenous data matrix)

    '''

    def __init__(self,errors,jacob,M):
        self.errors = errors
        self.jacob = jacob
        self.M = M
        self.n = errors.shape[0]
        self.p = errors.shape[1]

    def step(self):
        h = self.M.rho(self.errors)
        h = np.sqrt(h)
#        print h
        jacob = self.jacob
#        print np.linalg.pinv(self.tau(h,jacob))
#        print np.dot(np.linalg.pinv(self.tau(h,jacob)),self.lambd(h,jacob))
        return np.dot(np.linalg.pinv(self.tau(h,jacob)),self.lambd(h,jacob))

    def lambd(self,h,jacob):
        psi = np.transpose(self.M.psi(self.errors)).flatten('F')
        delt = self.n*[self.delta(h)]
        delt = scipy.linalg.block_diag(*delt)
        return np.dot(np.transpose(jacob),np.dot(delt,psi))

    def delta(self,h):
        n = self.n
        return (np.dot(np.transpose(h),h))/n

    def deltainverse(self,h):
#        print np.linalg.pinv(self.delta(h))
        return np.linalg.pinv(self.delta(h))

    def tau(self,h,jacob):
#        print jacob.shape
#        print self.u(h).shape
#        print self.u(h)
        return np.dot(np.transpose(jacob),np.dot(self.u(h),jacob))

    def u(self,h):
        temp = np.dot(self.deltainverse(h),self.w())
        temp = self.n*[temp]
        U = scipy.linalg.block_diag(*temp)
        return U

    def w(self):
        psi_deriv = self.M.psi_deriv(self.errors)
#        print psi_deriv
        p = self.p
        n = self.n
        gamma = np.zeros(p,dtype=np.float64)
        for i in range(p):
            gamma[i] = psi_deriv[:,i].sum()/float(n)
        return np.diag(gamma)
