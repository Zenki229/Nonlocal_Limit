#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import linalg
import matplotlib.pyplot as plt


# In[ ]:


def primat(M):
    for row in M:
        for item in row:
                print('%4f'% item,end='  ')
        print()


# In[ ]:


def legpol(x,N):
        P = np.zeros((N+1,np.size(x)))
        P[0] = np.ones((1,np.size(x)))
        P[1] = x
        dP = np.zeros((N,np.size(x)))
        for j in np.arange(1,N):
            P[j+1] = ((2*(j+1)-1)*x*P[j]-(j)*P[j-1])/(j+1);
            dP[j] = (j)*(x*P[j]-P[j-1])/(x**2-1)
        pN = P[N-1]
        dpN = dP[N-1]
        return pN, dpN


# In[ ]:


def gauleg(x1,x2,N):
    z = np.zeros(N)
    xm = 0.5*(x1+x2)
    xl = 0.5*(x2-x1)
    for n in np.arange(0,N):
        z[n] = math.cos((math.pi*(n+1-0.25))/(N+0.5))
        z1 = 100 * z[n]
        while abs(z1-z[n])> np.finfo(float).eps :
            (pN,dpN) = legpol(z[n],N+1)
            z1 = z[n]
            z[n] = z1-pN/dpN
    (pN,dpN) = legpol(z,N+1)
    x = xm - xl*z
    w = 2*xl/((1-z**2)*(dpN**2))
    x = x.T
    w = w.T
    return x, w


# In[ ]:


(x,w)= gauleg(0,1,10)


# In[ ]:


def Mass_Stiff_1D(vert,t,nf):

    nn = np.size(vert)
    nt = np.shape(t)[0]
    (node,W) = gauleg(0,1,20)
    Mass = np.zeros((nn,nn))
    Stiff = np.zeros((nn,nn))
    for i in np.arange(0,nt):
        x0 = vert[t[i][0]]
        x1 = vert[t[i][1]]
        x = x0+(x1-x0)*node
        aux = np.zeros((2,2))
        fy = np.zeros((2,np.size(x)))
        fy[0] = (x-x0)/(x1-x0)
        fy[1] = (x1-x)/(x1-x0)
        for j in [0,1]:
            for k in [0,1]:
                aux[j][k] = (x1-x0)*W.T@(fy[j]*fy[k])
        bux = 1/(x1-x0)*np.array([[1,-1],[-1,1]])
        Mass[np.ix_(t[i,:],t[i,:])] += aux
        Stiff[np.ix_(t[i,:],t[i,:])] += bux
    Mass = Mass[np.ix_(nf,nf)]
    Stiff = Stiff[np.ix_(nf,nf)]
    return Mass, Stiff


# In[ ]:


def proj_l2_1D(vert,t,nf,Mass,f):
    (node,W)= gauleg(0,1,20)
    nn = np.size(vert)
    nt = np.shape(t)[0]
    b = np.zeros(nn)
    def fy(n,x,x0,x1):
        if n == 0:
            result = (x1-x)/(x1-x0)
        if n == 1:
            result = (x-x0)/(x1-x0)
        return result
    for i in np.arange(0,nt):
        x0 = vert[t[i][0]]
        x1 = vert[t[i][1]]
        aux = np.zeros(2)
        x = (x1-x0)*node+x0
        for j in np.arange(0,2):
            aux[j] = (x1-x0)*np.dot(W,(f(x)*fy(1,x,x0,x1)))
        b[np.ix_(t[i][:])] += aux
    b = b[np.ix_(nf)]
    #fh = np.linalg.solve(Mass,b)
    fh = conjgrad(Mass,b)
    return fh
def conjgrad(A,b):
    tol = 1e-10
    x = b
    r = b- A.dot(x)
    if np.linalg.norm(r) < tol:
        return x
    y = -r
    z = A.dot(y)
    s = np.dot(y,z)
    t = np.dot(r,y)/s
    x = x+t*y
    for k in np.arange(0,np.size(b)):
        r = r-t*z
        if np.linalg.norm(r) < tol:
            return x
        B = np.dot(r,z)/s
        y = -r+B*y
        z = A.dot(y)
        s = np.dot(y,z)
        t = np.dot(r,y)/s
        x = x+t*y
    return x

# In[ ]:


(node,W)= gauleg(0,1,20)
M = 100;
h = 1/M;
vert = np.linspace(0,1,M+1).T
nn = np.size(vert)
t = np.block([[np.arange(0,nn-1)],[np.arange(1,nn)]]).T
nf = np.arange(1,nn-1).T
(Mass,Stiff) = Mass_Stiff_1D(vert,t,nf)
def f(x):
    result = x*(1-x)
    return result
fh = proj_l2_1D(vert,t,nf,Mass,f)
fig,ax=plt.subplots()
ax.plot(vert,np.concatenate([[0],fh,[0]]),linewidth=1.8)
ax.plot(vert,f(vert),linestyle=':')


# In[ ]:


def Checkratio(v,p):
    nn = np.size(v)-1
    Ratio = np.zeros(nn)
    for j in np.arange(0,nn):
        Ratio[j] = v[j]/v[j+1]
    Ratio = np.log(Ratio)/np.log(p[1:-1]/p[0:-2])
    return Ratio


# In[ ]:




