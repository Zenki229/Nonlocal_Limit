from lib import *
from scipy.integrate import quad
from Preparation_1D import *
# Generate 1D Stiffness matrix for nonlocal operator with fractional kernel
# Base on the paper of Tian, DU, 2013，SINUM，Link: https://epubs.siam.org/doi/10.1137/13091631X

def kernel_fun(x,s,delta):
    return (2-2*s)*x**(-1-2*s)*(x<=delta and x>=0)*delta**(2*s-2)/2
def  stiff_nonlocal_row_free(h,s,delta):
    N=int(1/h-1)
    r=int(np.floor(delta/h+0.0001))

    # r<=1
    if r<=1:
        pass
    # m=|i-j|=0
    ann1=quad(lambda t:kernel_fun(t,s,delta)*(-t**3/h**3+2*t**2/h**2),0,h)[0]
    ann2=quad(lambda t:kernel_fun(t,s,delta)*(t**3/(3*h**3)-2*t**2/h**2+4*t/h-4/3),h,2*h)[0]
    ann3=quad(lambda t:kernel_fun(t,s,delta),2*h,delta)[0]
    ann=(ann1+ann2+4*ann3/3)
    # m=1
    anplus1=quad(lambda t:kernel_fun(t,s,delta)*(2*t**3/(3*h**3)-t**2/h**2),0,h)[0]
    anplus2=quad(lambda t:kernel_fun(t,s,delta)*(-t**3/(2*h**3)+5*t**2/(2*h**2)-7*t/(2*h)+7/6),h,2*h)[0]
    anplus3=quad(lambda t:kernel_fun(t,s,delta)*(t**3/(6*h**3)-3*t**2/(2*h**2)+9*t/(2*h)-25/6),2*h,3*h)[0]
    anplus4=quad(lambda t:kernel_fun(t,s,delta),3*h,delta)[0]/3
    anplus = anplus1+anplus2+anplus3+anplus4
    # set r>=2
    andpls=np.zeros(r)
    for m in np.arange(2,r+2):
        andpls1=quad(lambda t:kernel_fun(t,s,delta)*(-t**3/(6*h**3)+(m-2)*t**2/(2*h**2)-(m**2-4*m+4)*t/(2*h)\
                                                     +(m**3-6*m**2+12*m-8)/6),(m-2)*h,(m-1)*h)[0]
        andpls2=quad(lambda t:kernel_fun(t,s,delta)*(t**3/(2*h**3)-(3*m-2)*t**2/(2*h**2)+(3*m**2-4*m)*t/(2*h)\
                                                     -(3*m**3-6*m**2+4)/6),(m-1)*h,m*h)[0]
        andpls3=quad(lambda t:kernel_fun(t,s,delta)*(-t**3/(2*h**3)+(3*m+2)*t**2/(2*h**2)-(3*m**2+4*m)*t/(2*h)\
                                                     +(3*m**3+6*m**2-4)/6),m*h,(m+1)*h)[0]
        andpls4=quad(lambda t:kernel_fun(t,s,delta)*(t**3/(6*h**3)-(m+2)*t**2/(2*h**2)+(m**2+4*m+4)*t/(2*h)\
                                                     -(m**3+6*m**2+12*m+8)/6),(m+1)*h,(m+2)*h)[0]
        andpls[m-2]=andpls1+andpls2+andpls3+andpls4
    row=np.zeros(r+2)
    for kj in np.arange(0,r+2):
        if kj==0:
            row[kj]=ann
        elif kj==1:
            row[kj]=anplus
        else: row[kj]=andpls[kj-2]
    if N-np.shape(row)[0]>0:
        row=h*np.concatenate((row,np.zeros(N-np.shape(row)[0])))
    else:
        row=h*row
    return row

def nonlocal_stiff_row_1D(Row,Node,delta,type)：


class Nonlocal_Model():
    def __init__(self,Node,Elem,FreeNode,h,s,delta):
        self.Mass,=Mass_Stiff_1D(Node,Elem,FreeNode)
        Row=stiff_nonlocal_row_free(h,s,delta)
        self.Stiff=nonlocal_stiff_row_1D(Row,Node,delta,'inout')




print(stiff_nonlocal_row_free(0.001,0.5,0.08))