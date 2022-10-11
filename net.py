from lib import *
class CNN(nn.Module):
    def __init__(self,N1,N2,NN,NL):
        super(CNN,self).__init__()
        self.N1=N1
        self.N2=N2
        self.NN=NN
        self.cn1=nn.Conv1d(in_channels=1,out_channels=NN,kernel_size=N1)
        self.ln=nn.Linear(NN,NN)
        self.cn2=nn.Conv1d(in_channels=1,out_channels=N2,kernel_size=NN)
        # self.ln2=nn.Linear(N2,N2)
        self.hid_cn1 = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=NN,kernel_size=NN)\
                                       for i in range(NL)])
        # self.hid_cn2=nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=N1,kernel_size=N2)\
        #                                for i in range(NL)])
    def forward(self,x):
        out=self.act(self.cn1(x))
        out=out.view(-1,1,self.NN)
        for i,cno in enumerate(self.hid_cn1):
            out=self.act(cno(out))
            out=out.view(-1,self.NN)
            out=self.act(self.ln(out))
            out=out.view(-1,1,self.NN)
        out=self.act(self.cn2(out))
        out=out.view(-1,self.N2)
        return out
    def act(self,x):
        return F.sigmoid(x)

class DNN(nn.Module):
    # NL: the number of hidden layers
    # NN: the number of vertices in each layer
    def __init__(self,N1,N2,NN, NL):
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(N1, NN)
        self.hidden_layers = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, N2)

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out

    def act(self, x):
        return F.relu(x)
