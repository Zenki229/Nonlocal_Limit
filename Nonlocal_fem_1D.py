from lib import *
from Matlab_Improve import *
from Preparation_1D import *
# Generate 1D Stiffness matrix for nonlocal operator with fractional kernel
# Base on the paper of Tian, DU, 2013，SINUM，Link: https://epubs.siam.org/doi/10.1137/13091631X

def kerf_local(x,s,delta):
    return (2-2*s)*x**(-1-2*s)*(x<=delta and x>=0)*delta**(2*s-2)/2
def kerf_frac(x,s,delta):
    return 4**s*s*math.gamma(s+1/2)/(math.pi**0.5*math.gamma(1-s))*x**(-1-2*s)*(x>=0 and x<=delta)
def Scale(type,s):
    if type=='local':
        aux=quad(lambda x:kerf_local(x,s,1)*x**2,0,1)[0]
    if type=='frac':
        aux = quad(lambda x: kerf_frac(x, s, 1) * x ** 2, 0, 1)[0]
    return 1/aux
def  stiff_nonlocal_row_free(h,s,delta,type):
    N=int(1/h-1)
    r=int(np.floor(delta/h+0.0001))
    if type=='local':
        kernel_fun=kerf_local
    elif type=='frac':
        kernel_fun=kerf_frac
    else:
        print('Error:Please input model type(frac or local)')
        pass
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

def nonlocal_stiff_row_1D(Scale,Row,Node,FreeNodeInd,TypeBd):
    Num_Node=Node.shape[0]
    Num_Row=Row.shape[0]
    Band=np.concatenate((Row[-1:0:-1],Row))
    StartInd=Row.shape[0]-1
    Stiff=np.zeros((Num_Node,Num_Node))
    for i in np.arange(Num_Node):
        head_global=max(0,i-StartInd)
        head_local=max(0,StartInd-i)
        tail_global=min(Num_Node-1,Num_Row+i-1)
        tail_local=StartInd+tail_global-i
        Stiff[i,head_global:tail_global+1]=Band[head_local:tail_local+1]
        Stiff[i,i]=-np.sum(Stiff[i,:])+Stiff[i,i]
    Stiff=Scale*Stiff
    if TypeBd=='Dirichlet':
        StiffDirichlet=np.eye(Num_Node)
        StiffDirichlet[FreeNodeInd.flatten(),:]=Stiff[FreeNodeInd.flatten(),:]
        return StiffDirichlet
    if TypeBd=='NeumannInOut':
        return Stiff
    if TypeBd=='NeumannIn':
        pass
    if TypeBd=='Robin':
        pass






class Nonlocal_Model():
    def __init__(self,Node,Elem,FreeNodeInd,h,s,delta,type_mod,type_Neumann):
        Num_Node=Node.shape[0]
        Idx=np.arange(Num_Node)
        self.Mass,Stiff=Mass_Stiff_1D(Node,Elem,Idx,lambda x:1)
        Row=stiff_nonlocal_row_free(h,s,delta,type_mod)
        self.Stiff=nonlocal_stiff_row_1D(Scale(type_mod,s),Row,Node,FreeNodeInd,type_Neumann)
        #if q!=0:
        #    self.QMass,Stiff=Mass_Stiff_1D(Node,Elem,FreeNodeInd,q)
        #elif q==0:
        #    self.QMass=np.zeros_like(self.Mass)








