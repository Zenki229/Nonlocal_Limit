from pde_model import *
Trainfile='train1.npy'
trainadd=NpyAppendArray(Trainfile)
s=0.5
delta =1
M=400
X=np.linspace(-delta,1+delta,4*M+1)
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
Count=100
# data_DtN=np.zeros((3,BdNodeInd.size))
for Epoch in np.arange(Count):
    Num_Potential=3
    sigma = np.ones(Num_Potential)*0.02
    Center=np.array([0.1, 0.5, 0.9])
    # Center = np.random.random(size=Num_Potential) * 0.9 + 0.05
    # Scale =np.array([0.5, 1, 3])
    Scale = np.random.rand(Num_Potential)*1000
    def Potential(x):
        aux = 0
        for i in np.arange(Num_Potential):
            aux+=Scale[i]*np.exp(-0.5*((x-Center[i])/sigma[i])**2)
        return aux
    Pot_Info = Potential(Node[FreeNodeInd.flatten()]).reshape(1,-1)
    QMass,Stiff=Mass_Stiff_1D(Node,Elem,FreeNodeInd,Potential)
    A=np.zeros_like(T.Stiff)
    A[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]+=QMass
    A+=StiffDirichlet
    U0 = np.linalg.solve(A,pde_model.RHS.flatten())
    # plt.plot(Node.flatten(),U0.flatten())

    #Compute DtN
    Mass=T.Mass[np.ix_(BdNodeInd.flatten(),BdNodeInd.flatten())]
    DtN_Weak=np.dot(T.Stiff,U0.flatten())
    DtN = np.linalg.solve(T.Mass, DtN_Weak.flatten())[BdNodeInd.flatten()].reshape(1,-1)
    # DtN = DtN[len(LBdNodeInd)+np.array([-2,0,1,2])].reshape(1,-1)
    # data_DtN[Epoch,:]=DtN
    # plt.plot(DtN.flatten())
    aux=np.concatenate((Pot_Info,DtN),axis=1)
    # np.savetxt('data_DtN',data_DtN)
    trainadd.append(aux)
# plt.show()