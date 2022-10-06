from lib import *
from Nonlocal_fem_1D import *

M=5
delta = 0.5

vert=np.linspace(0,1,2*M)
Left_vert=np.linspace(-delta,0,M)
Right_vert=np.linspace(1,delta,M)
X=np.linspace(-delta,1+delta,4*M+1)
vert=np.where(X>0)
print((X>0) and (X<1))