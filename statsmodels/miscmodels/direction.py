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

    def __call__(self):
        h = self.M.rho(self.errors)
        jacob = self.jacob
        return np.dot(np.pinv(tau(h,jacob)),lambd(h,jacob))

    def lambd(h,jacob):
        psi = self.M.psi(errors)
        delt = self.p*[delta(h)]
        delt = scipy.linalg.block_diag(*delt)
        return np.dot(jacob,np.dot(delt,psi))

    def delta(self,h):
        return (np.dot(np.transpose(h),h))/n

    def deltainverse(self,h):
        return np.pinv(delta(h))

    def tau(self,h,jacob):
        return np.dot(np.transpose(jacob),np.dot(u(h),
               np.dot(w,jacob)))
    def u(h):
        temp = np.dot(deltainverse(h),w(psi_deriv))
        temp = p*[temp]
        U = scipy.linalg.block_diag(*temp)
        return U

    def w(self):
        psi_deriv = self.M.psi_deriv(self.errors)
        gamma = np.zeros(self.p)
        for i in range(p):
            gamma[i] = psi_deriv[i].sum()/n
        return np.diag(gamma)
