from lib import *
def primat(M):
    for row in M:
        for item in row:
                print('%4f'% item,end='  ')
        print()
def legpol(x,N):
        P = np.zeros((N+1,np.size(x)))
        P[0] = np.ones((1,np.size(x)))
        P[1] = x
        dP = np.zeros((N,np.size(x)))
        for j in np.arange(1,N):
            P[j+1] = ((2*(j+1)-1)*x*P[j]-(j)*P[j-1])/(j+1);
            dP[j] = (j)*(x*P[j]-P[j-1])/(x**2-1)
        pN = P[N-1]
        dpN = dP[N-1]
        return pN, dpN
def gauleg(x1,x2,N):
    z = np.zeros(N)
    xm = 0.5*(x1+x2)
    xl = 0.5*(x2-x1)
    for n in np.arange(0,N):
        z[n] = math.cos((math.pi*(n+1-0.25))/(N+0.5))
        z1 = 100 * z[n]
        while abs(z1-z[n])> np.finfo(float).eps :
            (pN,dpN) = legpol(z[n],N+1)
            z1 = z[n]
            z[n] = z1-pN/dpN
    (pN,dpN) = legpol(z,N+1)
    x = xm - xl*z
    w = 2*xl/((1-z**2)*(dpN**2))
    x = x.T
    w = w.T
    return x, w
def fy(type, x, x0, x1):
    if type == 0:
        return (x - x0) / (x1 - x0)
    elif type == 1:
        return (x1 - x) / (x1 - x0)
    else:
        return "error Mass_Stiff"
    # Higherorder
def Mass_Stiff_1D(Node,Elem,FreeNodeInd,q):
    nn = Node.shape[0]
    nt = Elem.shape[0]
    Mass = np.zeros((nn,nn))
    Stiff = np.zeros((nn,nn))
    for i in np.arange(nt):
        x0 = Node[Elem[i,0]]
        x1 = Node[Elem[i,1]]
        aux = np.zeros((2,2))
        for j in [0,1]:
            for k in [0,1]:
                aux[j,k] = quad(lambda x:q(x)*fy(j,x,x0,x1)*fy(k,x,x0,x1),x0,x1)[0]
        bux = 1/(x1-x0)*np.array([[1,-1],[-1,1]])
        Mass[np.ix_(Elem[i,:],Elem[i,:])] += aux
        Stiff[np.ix_(Elem[i,:],Elem[i,:])] += bux
    Mass = Mass[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]
    Stiff = Stiff[np.ix_(FreeNodeInd.flatten(),FreeNodeInd.flatten())]
    return Mass, Stiff
def proj_l2_1D(vert,t,nf,Mass,f):
    (node,W)= gauleg(0,1,20)
    nn = np.size(vert)
    nt = np.shape(t)[0]
    b = np.zeros(nn)
    def fy(n,x,x0,x1):
        if n == 0:
            result = (x1-x)/(x1-x0)
        if n == 1:
            result = (x-x0)/(x1-x0)
        return result
    for i in np.arange(0,nt):
        x0 = vert[t[i][0]]
        x1 = vert[t[i][1]]
        aux = np.zeros(2)
        x = (x1-x0)*node+x0
        for j in np.arange(0,2):
            aux[j] = (x1-x0)*np.dot(W,(f(x)*fy(1,x,x0,x1)))
        b[np.ix_(t[i][:])] += aux
    b = b[np.ix_(nf)]
    fh = conjgrad(Mass,b)
    return fh
def conjgrad(A,b):
    tol = 1e-10
    x = b
    r = b- A.dot(x)
    if np.linalg.norm(r) < tol:
        return x
    y = -r
    z = A.dot(y)
    s = np.dot(y,z)
    t = np.dot(r,y)/s
    x = x+t*y
    for k in np.arange(0,np.size(b)):
        r = r-t*z
        if np.linalg.norm(r) < tol:
            return x
        B = np.dot(r,z)/s
        y = -r+B*y
        z = A.dot(y)
        s = np.dot(y,z)
        t = np.dot(r,y)/s
        x = x+t*y
    return x
def Checkratio(v,p):
    nn = np.size(v)-1
    Ratio = np.zeros(nn)
    for j in np.arange(0,nn):
        Ratio[j] = v[j]/v[j+1]
    Ratio = np.log(Ratio)/np.log(p[1:-1]/p[0:-2])
    return Ratio

