from Nonlocal_fem_1D import *
Datafile = 'data_train.npy'
Potentialfile='Pot_train.npy'
DataAdd = NpyAppendArray(Datafile)
PotAdd=NpyAppendArray(Potentialfile)
s=0.75
delta = 0.5
M=100
X=np.linspace(-delta,1+delta,4*M+1)
Vert=X[(X>0)&(X<1)]
Left_Vert=X[X<=0]
Right_Vert=X[X>=1]
Node=X.T
Num_Node=Node.shape[0]
Elem=np.stack((np.arange(Num_Node-1).T,np.arange(1,Num_Node).T),axis=1)
FreeNodeInd=np.asarray(np.where((X>0)&(X<1)))
BdNodeInd=np.asarray(np.where((X<=0)|(X>=1)))
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
Count=100000
for Epoch in np.arange(Count):
    Num_Potential=4
    sigma = np.random.random(size=Num_Potential) * 0.9 + 0.05
    Center = np.random.random(size=Num_Potential) * 0.9 + 0.05
    Scale = np.random.random(size=Num_Potential) * 0.1 - 0.05
    Pot_Info = np.concatenate((sigma, Center, Scale)).reshape(1,-1)
    def Potential(x):
        aux = 0
        for i in np.arange(Num_Potential):
            aux+=Scale[i]*np.exp(-0.5*((x-Center[i])/sigma[i])**2)
        return aux+1
    QMass,Stiff=Mass_Stiff_1D(Node,Elem,FreeNodeInd,Potential)
    A=B
    A[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]+=QMass
    U0 = np.linalg.solve(A,RHS.flatten())
    #Compute DtN
    Mass=T.Mass[np.ix_(BdNodeInd.flatten(),BdNodeInd.flatten())]
    DtN_Weak=np.dot(T.Stiff,U0.flatten())[BdNodeInd]
    DtN = np.linalg.solve(Mass, DtN_Weak.flatten()).reshape(1,-1)
    PotAdd.append(Pot_Info)
    DataAdd.append(DtN)