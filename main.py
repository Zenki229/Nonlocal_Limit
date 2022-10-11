import torch.utils.data
from Nonlocal_fem_1D import *
from net import *
Trainsets=np.load('train.npy')
BATCH_SIZE=int(Trainsets.shape[0]*0.02)
train_loader=torch.utils.data.DataLoader(Trainsets,batch_size=BATCH_SIZE,shuffle=True)
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
model=CNN(N2,N1,N1,3)
lr=0.00001
epsi=0.00001
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
lossF=nn.MSELoss()
Epoch=50
for epoch in np.arange(Epoch):
    for enum,XY in enumerate(train_loader):
        X=XY[:,:N1].reshape(-1,1,N1).to(torch.float32)
        Y=XY[:,N1:].reshape(-1,1,N2).to(torch.float32)
        out = model((Y-DtN0))/U00
        loss = torch.mean(torch.sqrt(torch.sum((out - X) ** 2, dim=1) / N2))
        # X = XY[:, :N1].to(torch.float32)
        # Y = XY[:, N1:].to(torch.float32)
        # out = model((Y-DtN0))/U00
        # loss = torch.mean(torch.sqrt(torch.sum((out - X) ** 2, dim=1) / N2))
        # # loss=lossF(out.to(torch.double),Y.to(torch.double))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{Epoch}], Loss: {loss.item():.8f}')
joblib.dump(model, 'my_trained_model.pkl', compress=0)
# Epoch = 10
# for epoch in np.arange(Epoch):
#     XY = next(iter(train_loader))
#     X = XY[:, :N]
#     Y = XY[:, N:]
#     out = Model(X)
#     loss = lossF(out, Y)
#     print (f'Epoch [{epoch+1}/{Epoch}], Loss: {loss.item():.4f}')