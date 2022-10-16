import torch

from lib import *
class CNN(nn.Module):
    def __init__(self,N1,N2,NN,NL):
        super(CNN,self).__init__()
        self.N1=N1
        self.N2=N2
        self.NN=NN
        self.cn1=nn.Conv1d(in_channels=N1,out_channels=NN,kernel_size=1)
        self.ln=nn.Linear(NN,NN)
        self.cn2=nn.Conv1d(in_channels=NN,out_channels=N2,kernel_size=1)
        self.ln2=nn.Linear(N2,N2)
        self.hid_cn1 = nn.ModuleList([nn.Conv1d(in_channels=NN,out_channels=NN,kernel_size=1)\
                                       for i in range(NL)])
        # self.hid_cn2=nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=N1,kernel_size=N2)\
        #                                for i in range(NL)])
    def forward(self,x):
        out=self.act(self.cn1(x))
        out=out.view(-1,self.NN,1)
        for i,cno in enumerate(self.hid_cn1):
            out=self.act(cno(out))
            # out=out.view(-1,self.NN)
            # out=self.act(self.ln(out))
            # out=out.view(-1,self.NN,1)
        out=self.act(self.cn2(out))
        out=out.view(-1,self.N2)
        out=self.ln2(out)
        return out
    def act(self,x):
        return F.relu(x)

class CNN2(nn.Module):
    def __init__(self,N1,N2,NN,NL):
        super(CNN2,self).__init__()
        self.N1=N1
        self.N2=N2
        self.NN=NN
        self.cn1=nn.Conv1d(in_channels=1,out_channels=NN,kernel_size=N1)
        self.ln=nn.Linear(NN,NN)
        self.cn2=nn.Conv1d(in_channels=1,out_channels=N2,kernel_size=NN)
        self.ln2=nn.Linear(N2,N2)
        self.hid_cn1 = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=NN,kernel_size=NN)\
                                       for i in range(NL)])
        # self.hid_cn2=nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=N1,kernel_size=N2)\
        #                                for i in range(NL)])
    def forward(self,x):
        out=self.act(self.cn1(x))
        out=out.view(-1,1,self.NN)
        for i,cno in enumerate(self.hid_cn1):
            out=self.act(cno(out))
            out=out.view(-1,1,self.NN)
            # out=self.act(self.ln(out))
            # out=out.view(-1,self.NN,1)
        out=self.act(self.cn2(out))
        out=out.view(-1,self.N2)
        out=self.ln2(out)
        return out
    def act(self,x):
        return F.relu(x)
