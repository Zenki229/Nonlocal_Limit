from lib import *
from Nonlocal_fem_1D import *
class PDE():
    def __init__(self,Node,Elem,FreeNodeInd,BdNodeInd,LBdNodeInd,RBdNodeInd,FEM_structure,DirichletFunc_Left,DirihchletFunc_Right,source):
        RHS = np.zeros_like(Node)
        RHS[BdNodeInd] = np.concatenate((DirichletFunc_Left(Node[LBdNodeInd]), DirihchletFunc_Right(Node[RBdNodeInd])), axis=1)
        mass, stiff = Mass_Stiff_1D(Node,Elem,FreeNodeInd, lambda x: 1)
        source_weak = proj_l2_1D(Node,Elem,FreeNodeInd,mass,source)
        RHS[FreeNodeInd]=source_weak
        self.RHS=RHS
