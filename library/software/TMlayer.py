import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod

class TMlayer(nn.Linear):
    """Fully connected layer class object using the Forward Forward method
    """
    
    def __init__(self, shape_in, shape_out, TM, bias=False, device=None, requires_grad=True, dtype=torch.cfloat, last_layer=False):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param bias: If True, adds a learnable bias to the output. Default; True
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated.
        :return: Forward Forward fully connected linear layer object
        :rtype: nn.Module
        """
        [self.Nix, self.Niy], [self.Nox, self.Noy] = shape_in, shape_out 
        in_features, out_features =  self.Nix* self.Niy, self.Nox*self.Noy # = self.Nix*self.Niy, self.Nox*self.Noy
        self.TM = torch.from_numpy(TM).to(device).type(dtype)
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.w0 = torch.Tensor(in_features).uniform_(to=2*np.pi).to(device) #storing initial weight
        self.weight = nn.Parameter(self.w0.clone())
        self.dtype = dtype

    def forward(self, x):
        """Applies a linear transformation to the inconming data: $y = x W^T + b$

        The input x is first normalized so that we don't biais the following layers and we focus on the input vector direction only.

        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: Linear transformation of the input normalized x by the weight matrix 
        :rtype: torch.Tensor
        """
        x = x.type(self.dtype)
        nb, nx, ny = x.shape[0], x.shape[-2], x.shape[-1]
        x = x[:nb, : ,nx//2-self.Nix//2:nx//2+self.Nix//2, ny//2-self.Niy//2:ny//2+self.Niy//2 ]#Crop brutal
        self.x = x
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = x.reshape(x.shape[0], 1, self.in_features)*torch.exp(1j*self.weight)
        self.x_direction = x_direction
        x_out = torch.matmul(x_direction, self.TM).reshape(x.shape[0],1,self.Nox,self.Noy)
        #x_out = torch.stack([torch.matmul(self.TM, x_direction[i,::]).reshape(self.Nox,self.Noy)[None,::] for i in range(x.shape[0])])
        self.x_out = x_out
        return x_out #+ self.bias.unsqueeze(0) TODO check this out

class CameraModSquare(nn.Module):
    """[Summary]

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    
    def __init__(self,center=True,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.training=False
        
    def forward(self,x):
        return torch.abs(x)**2#/torch.sum(torch.abs(x+self.eps*torch.ones_like(x))**2)