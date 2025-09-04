import torch.nn as nn
from models import register

@register('mlp')
class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden_list):
        super().__init__()
        layers=[]
        lastv=in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv,hidden))
            layers.append(nn.ReLU())
            lastv=hidden
        layers.append(nn.Linear(lastv,out_dim))
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        shape=x.shape[:-1]
        x=self.layers(x.view(-1,x.shape[-1]))
        return x.view(*shape,-1)

if __name__=='__main__':
    import torch
    model=MLP(in_dim=64,out_dim=128,hidden_list=[64,64,64])
    x=torch.randn(10,3,4,4,4,23,64)
    y=model(x)
    print(y.shape)