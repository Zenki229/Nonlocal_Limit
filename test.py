from lib import *
from Nonlocal_fem_1D import *

M=50
delta = 0.5
X=np.linspace(-delta,1+delta,4*M+1)
Vert=X[(X>0)&(X<1)]
Left_Vert=X[X<=0]
Right_Vert=X[X>=1]
Node=X.T
s=0.75
Num_Node=Node.shape[0]
Elem=np.stack((np.arange(Num_Node-1).T,np.arange(1,Num_Node).T),axis=1)
FreeNodeInd=np.asarray(np.where((X>0)&(X<1)))
BdNodeInd=np.asarray(np.where((X<=0)|(X>=1)))
LBdNodeInd=np.asarray(np.where(X<=0))
RBdNodeInd=np.asarray(np.where(X>=1))
h=1/(2*M)
type_mod='frac'
type_Neumann='NeumannInOut'
def Potential(x):
    return np.sin(np.pi*x)+0.1
def DirichletFunc_Left(x):
    return x
def DirihchletFunc_Right(x):
    return x-1
T=Nonlocal_Model(Node,Elem,FreeNodeInd,Potential,h,s,delta,type_mod,type_Neumann)
U0=np.zeros_like(Node)
U0[BdNodeInd]=np.concatenate((DirichletFunc_Left(Node[LBdNodeInd]),DirihchletFunc_Right(Node[RBdNodeInd])),axis=1)
RHS=np.dot(T.Stiff,U0.flatten())
RHS=RHS[FreeNodeInd]
A=T.Stiff[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]+T.QMass
U0[FreeNodeInd]=np.linalg.solve(A,RHS.flatten())
DtN=np.dot(T.Stiff,U0.flatten())[BdNodeInd]
plt.plot(Node,U0)
plt.show()
with open("foo.csv",'w',newline='') as file:
    writer=csv.writer(file,delimiter=",")
    for i in DtN:
        writer.writerow(i)

