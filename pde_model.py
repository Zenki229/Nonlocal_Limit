from lib import *
from Nonlocal_fem_1D import *
class PDE():
    def __init__(self,Node,FreeNodeInd,BdNodeInd,LBdNodeInd,RBdNodeInd,FEM_structure,DirichletFunc_Left,DirihchletFunc_Right):
        RHS = np.zeros_like(Node)
        RHS[BdNodeInd] = np.concatenate((DirichletFunc_Left(Node[LBdNodeInd]), DirihchletFunc_Right(Node[RBdNodeInd])), axis=1)
        self.source=RHS
