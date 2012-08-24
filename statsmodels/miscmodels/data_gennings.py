import numpy as np
from statsmodels.miscmodels.rnlm import RNLM
import statsmodels.miscmodels.norms as norms
import statsmodels.api as sm 

x1=np.array([0,1,2.5,5])
x1 = x1.repeat(16)
x2 = np.array([0,5,10,25])
x2 = np.tile(x2.repeat(4),4)
t = np.array([0.01,.25,.5,1.0,2.0,3.0])
rows= 64
cols = 6
exog_shape = (rows,cols,8)
exog = np.empty(exog_shape)
for i in range(rows):
    for j in range(cols):
            exog[i][j][0] = 1.0
            exog[i][j][1] = x1[i]
            exog[i][j][2] = x2[i]
            exog[i][j][3] = t[j]
            exog[i][j][4] = x1[i]*x2[i]
            exog[i][j][5] = x1[i]*t[j]
            exog[i][j][6] = x2[i]*t[j]
            exog[i][j][7] = x1[i]*x2[i]*t[j]

b1 = -2.1828
b2 = 0.0695
b3 = 0.0066
b4 = 0.0946
b5 = -0.0052
b6 = 0.0777
b7 = 0.0259
b8 = -0.0022
#b1=b2=b3=b4=b5=b6=b7=b8=1

start=np.array([b1,b2,b3,b4,b5,b6,b7,b8],dtype = np.float64)
x=np.array([9,10,8,8,11,14,7,6,11,14,7,5,10,11,7,6,11,14,8,9,13,16,10,8,10,14,7,
            6,9,11,6,5,9,10,8,10,10,11,8,7,12,14,9,9,7,9,5,4,9,9,8,10,11,12,10,
            8,10,11,9,5,7,9,5,4],
[9,10,8,6,14,16,13,7,20,24,17,15,25,33,17,16,13,15,10,8,18,22,14,15,
            25,26,24,16,23,28,15,15,19,19,18,21,22,24,19,21,28,23,33,19,22,24,
            21,15,52,60,45,42,21,21,20,17,24,23,25,12,21,23,23,12],
[8,9,8,7,12,18,13,9,36,27,18,16,51,39,24,24,9,14,10,9,37,22,16,18,61,
            30,27,21,39,40,22,27,56,21,19,23,57,28,23,25,33,37,26,23,59,31,29,
            25,77,60,50,44,27,22,22,21,26,27,29,16,55,31,31,20],
[10,12,9,8,14,20,14,11,46,29,21,19,65,48,34,31,10,16,11,11,41,29,16,
            19,57,30,29,24,58,42,30,36,64,23,19,28,62,30,24,28,43,43,31,29,65,
            35,36,36,78,57,49,62,30,27,24,26,33,31,33,22,60,41,35,31],
[10,15,9,10,13,21,15,12,44,32,22,22,66,52,37,36,11,19,12,12,42,30,20,21,
            60,35,32,27,53,75,44,43,33,28,21,29,66,35,27,30,49,47,34,34,67,46,
            54,40,73,73,60,62,36,32,28,27,39,36,37,27,66,58,53,41],
[12,13,10,11,14,21,16,12,46,34,22,23,70,55,41,41,11,21,13,13,46,21,
            18,21,63,29,32,27,67,72,56,55,34,23,20,31,70,30,31,32,58,40,36,34,
            67,45,72,48,76,79,71,73,41,28,33,32,47,31,40,29,66,67,66,57])

#print a.size,b.size,c.size,d.size,e.size,f.size

endog=np.column_stack([a, b, c, d, e, f])
endog = endog*0.01
ts = 1.5*np.std(endog,axis=0,dtype=np.float64)
ts = np.atleast_2d(ts)
ts = ts.repeat(rows,axis=0)
#print ts
class gennings(RNLM):

    def expr(self,params,exog):
#        func= np.empty((rows,cols))
#        for i in range(rows):
#            for j in range(cols):
#                temp = np.dot(exog[i][j],np.transpose(params))
#                func[i][j] = 1/(1+np.exp(-temp))
        func = np.dot(exog,np.transpose(params))
        func = 1/(1+np.exp(-func))
        return func

    def jacobian(self,params):
        jac= np.empty((rows*cols,8),dtype=np.float64)
        temp = np.dot(exog,np.transpose(params))
        temp = np.exp(-temp)/(1+np.exp(-temp)**2)
        for i in range(rows):
            for j in range(cols):
                jac[i*j] = temp[i][j]*exog[i][j]
        return jac

    def start_value(self):
        return start
'''        endog = self.endog
        linearised_endog = np.log(endog/(1-endog))
        start_val = np.empty(8)
        for i in range(6):
            mod = sm.OLS(linearised_endog[:,i],exog[:,i,:]).fit()
            start_val += mod.params
        return start_val/6
'''
m = norms.HuberT(t=ts)
mod = gennings(endog,exog,M=m)
mod.fit()
