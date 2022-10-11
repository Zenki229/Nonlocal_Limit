import torch.utils.data
from Nonlocal_fem_1D import *
from net import *
Model= joblib.load('my_trained_model.pkl')
testsets=np.load('test.npy')
BATCH_SIZE=1
test_loader=torch.utils.data.DataLoader(testsets,batch_size=BATCH_SIZE,shuffle=True)
s=0.75
delta = 0.5
M=100
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
type_Neumann='NeumannInOut'
def DirichletFunc_Left(x):
    return np.exp(-x)
def DirihchletFunc_Right(x):
    return np.exp(x)-1
T=Nonlocal_Model(Node,Elem,FreeNodeInd,h,s,delta,type_mod,type_Neumann)
RHS = np.zeros_like(Node)
RHS[BdNodeInd] = np.concatenate((DirichletFunc_Left(Node[LBdNodeInd]), DirihchletFunc_Right(Node[RBdNodeInd])), axis=1)
B = T.Stiff
B[BdNodeInd.flatten(), :] = 0
B[BdNodeInd, BdNodeInd] = 1
def Potential(x):
    return np.ones_like(x)
Pot_Info = Potential(Node[FreeNodeInd.flatten()]).reshape(1,-1)
QMass,Stiff=Mass_Stiff_1D(Node,Elem,FreeNodeInd,Potential)
A=B
A[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]+=QMass
U0 = np.linalg.solve(A,RHS.flatten())
U00=torch.from_numpy(U0[FreeNodeInd])
#Compute DtN
Mass=T.Mass[np.ix_(BdNodeInd.flatten(),BdNodeInd.flatten())]
DtN_Weak=np.dot(T.Stiff,U0.flatten())[BdNodeInd]
DtN0 = np.linalg.solve(Mass, DtN_Weak.flatten()).reshape(1,-1)
DtN0=torch.from_numpy(DtN0).to(torch.float32)
XY=next(iter(test_loader))
# print(XY)
q = XY[:, :N1].to(torch.float32)
Y = XY[:, N1:].to(torch.float32)
qNN=Model((Y-DtN0))/U00
plt.plot(X[FreeNodeInd].flatten(),q.flatten().detach().numpy(),X[FreeNodeInd].flatten(),qNN.flatten().detach().numpy())
plt.show()