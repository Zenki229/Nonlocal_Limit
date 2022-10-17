import torch.utils.data
from Nonlocal_fem_1D import *
from net import *
from pde_model import *
Model= joblib.load('train_model.pkl')
testsets=np.load('train1.npy')
BATCH_SIZE=1
test_loader=torch.utils.data.DataLoader(testsets,batch_size=BATCH_SIZE,shuffle=True)
s=0.5
delta = 1
M=400
X=np.linspace(-delta,1+delta,4*M+1)
Node=X.T
Num_Node=Node.shape[0]
Elem=np.stack((np.arange(Num_Node-1).T,np.arange(1,Num_Node).T),axis=1)
FreeNodeInd=np.asarray(np.where((X>0)&(X<1)))
BdNodeInd=np.asarray(np.where((X<=0)|(X>=1)))
N1=FreeNodeInd.size
N2=BdNodeInd.size
LBdNodeInd=np.asarray(np.where(X<=0))
RBdNodeInd=np.asarray(np.where(X>=1))
h=1/(2*M)
type_mod='frac'
type_Neumann='Dirichlet'
def DirichletFunc_Left(x):
    return 3*x**2/100
def DirihchletFunc_Right(x):
    return 2*(x-1)**2/100
def source(x):
    # return 10*np.sin(np.pi*x)
    return np.ones_like(x)*10
T=Nonlocal_Model(Node,Elem,FreeNodeInd,h,s,delta,type_mod,type_Neumann)
StiffDirichlet=np.eye(Num_Node)
StiffDirichlet[FreeNodeInd.flatten(),:]=T.Stiff[FreeNodeInd.flatten(),:]
pde_model=PDE(Node,Elem,FreeNodeInd,BdNodeInd,LBdNodeInd,RBdNodeInd,T,DirichletFunc_Left,DirihchletFunc_Right,source)
def Potential(x):
    return np.zeros_like(x)
QMass,Stiff=Mass_Stiff_1D(Node,Elem,FreeNodeInd,Potential)
A=np.zeros_like(T.Stiff)
A[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]+=QMass
A+=StiffDirichlet
U0 = np.linalg.solve(A,pde_model.RHS.flatten())
U00=torch.from_numpy(U0[FreeNodeInd])
#Compute DtN
Mass=T.Mass[np.ix_(BdNodeInd.flatten(),BdNodeInd.flatten())]
MassFree=T.Mass[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]
DtN_Weak=np.dot(T.Stiff,U0.flatten())[BdNodeInd]
DtN0 = np.linalg.solve(Mass, DtN_Weak.flatten()).reshape(1,-1)
DtN0=torch.from_numpy(DtN0).to(torch.float32)
XY=next(iter(test_loader))
# print(XY)
q = XY[:, :N1].to(torch.float32)
Y = XY[:, N1:].to(torch.float32).reshape(-1,1,N2)
qNN=Model((Y-DtN0.reshape(-1,1,N2)))/U00+torch.zeros_like(q)
err = torch.sqrt(torch.sum((q-qNN)**2))/torch.sqrt(torch.sum(q**2))
print(f'err= {err}')
plt.plot(X[FreeNodeInd].flatten(),q.flatten().detach().numpy(),X[FreeNodeInd].flatten(),qNN.flatten().detach().numpy())
plt.show()