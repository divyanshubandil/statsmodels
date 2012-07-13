'''
Robust Non-linear Regression
M-estimation of Nonlinear Models

'''

import numpy as np

from statsmodels.miscmodels.nonlinls import NonLinearModel
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import RLMResults
from statsmodels.robust.norms as norms

from direction import Step


class RNLM(NonLinearModel):
    '''
    Robust Nonlinear Models

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    Parameters
    ----------
    endog : array-like
        n x p endogenous response variable
    exog : array-like
        n x p x m exogenous design matrix
    M : statsmodels.robust.norms.RobustNorm, optional

    Notes
    -----

    Base class for M-estimation of non-linear models with Newton-Raphson method

    The algorithm takes care of multivariate data

    '''
    def __init__(self, endog=None, exog=None, M=norms.HuberT()):
        self.endog = endog
        self.exog = exog

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return self.expr(params, exog)

    def geterrors(self, params, weights=None):
        return (self.endog - self.predict(params, self.exog))

    def fit(self,maxiter=50):
        old_params = self.startvalue(params)
        params = None
        for iteration in range(maxiter):
            params = predict_params(old_params)
            if (check_convergence(params)):
                break
            self._store_params(params)
            old_params = params

        if iteration is maxiter:
            print 'maximum iterations completed.convergence not achieved'

    def start_value(self):
        from statsmodels.miscmodels.nonlinls import NonLinearLS
        start = NonlinearLS(self.endog,self.exog).fit()
        return start.params    

    def _check_convergence(params):
        pass

    def predict_params(self,old_params):
        return old_params + Step(self.geterrors,
                                     self.getjacobian(old_params),self.M)

    def jac_predict(self, params):
        '''jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        '''
        from statsmodels.sandbox.regression.numdiff \
             import approx_fprime_cs

        jaccs_err = approx_fprime_cs(params, self._predict)
        return jaccs_err

    def approx_jac_predict(self, params):
        '''approximate jacobian estimation
        
        Objective is to implement a better method for calculation of derivatives 
        than forward differences approach.
        eg- Automatic derivative, n-point numerical derivative
        
        We would like to give the user the option to give the jacobian of the
        function. scipy.optimize based on minpack encourages to do so.
        
        '''
        #Calculating the jacobian
        func = self.geterrors
        x = np.asarray(params)
        fx = func(x)
        jacob = np.zeros((len(np.atleast_1d(fx)),len(x)), float)
        inf = np.zeros((len(x),), float)
        h = 1e-10#seems to be the best value after running the test suite
        for i in range(len(x)):
            inf[i] = h
            jacob[:,i] = (func((x+inf)) - fx)/h
            inf[i] = 0.0
        return jacob

    def _store_params(self, params):
        ''' The parameter values calculated at each iteration of the algorithm is 
            stored for keeping in regression results
        '''
        params = np.array(params)
        if self.params_iter==None:
            self.params_iter=[params]
        else:
            self.params_iter.append(params)

    def getjacobian(self,params):
        '''The function to select the jacobian calculating function and return
        jacobian matrix received
        '''

        self._store_params(params)
        try:
            jac_func = -self.whiten(self.jacobian(params))
        except NotImplementedError:
            jac_func = self.approx_jac_predict(params)
        return jac_func

