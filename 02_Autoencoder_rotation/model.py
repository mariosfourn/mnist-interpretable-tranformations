import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class Net(nn.Module):
    """
    Autoencoder module for intepretable transformations
    """
    def __init__(self, device):
        super(Net, self).__init__()

        self.device = device

        self.encoder=Encoder(self.device)

        self.decoder=Decoder(self.device)
        
    def forward(self, x, params):
        #Encoder 
        x=self.encoder(x)

        #Feature transform layer
        x=feature_transformer(x, params,self.device)

        #Decoder
        x=self.decoder(x)
          
        return x


def feature_transformer(input, params,device):
    """For now we assume the params are just a single rotation angle

    Args:
        input: [N,c] tensor, where c = 2*int
        params: [N,1] tensor, with values in [0,2*pi)
    Returns:
        [N,c] tensor
    """
    # First reshape activations into [N,c/2,2,1] matrices
    x = input.view(input.size(0),input.size(1)//2,2,1)
    # Construct the transformation matrix
    sin = torch.sin(params)
    cos = torch.cos(params)
    transform = torch.cat([cos, -sin, sin, cos], 1)
    transform = transform.view(transform.size(0),1,2,2).to(device)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


def inverse_feature_transformer(input, params,device):
    """For now we assume the params are just a single rotation angle

    Args:
        input: [N,c] tensor, where c = 2*int
        params: [N,1] tensor, with values in [0,2*pi)
        device
    Returns:
        [N,c] tensor
    """
    # First reshape activations into [N,c/2,2,1] matrices
    x = input.view(input.size(0),input.size(1)//2,2,1)
    # Construct the transformation matrix
    sin = torch.sin(params)
    cos = torch.cos(params)
    #The inverse of a rotation matrix is its transpose
    transform = torch.cat([cos, sin, -sin, cos], 1)
    transform = transform.view(transform.size(0),1,2,2).to(device)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


class Net_Reg(nn.Module):
    """Autoencoder with regularisatoon """
    def __init__(self, device):
        super(Net_Reg, self).__init__()

        self.device=device

        self.encoder=Encoder(self.device)
        self.decoder=Decoder(self.device)

    def forward(self,x,y,params):
        """
        Args:
            x:      untransformed images pytorch tensor
            y:      transforedm images  pytorch tensor
            params: rotations
        """
        #Encoder 
        f=self.encoder(x) #feature vector for original image

        f_theta=inverse_feature_transformer(self.encoder(y), params,self.device) # feature vector for tranformed image

        #Feature transform layer
        x=feature_transformer(f, params,self.device)

        #Decoder
        x=self.decoder(x)

        #Return reconstructed image, feature vector of oringial image, feature vector of transformation

        return x, f, f_theta


# class Angle_Discriminator(nn.Module):
#     def __init__(self, device):
#         super(Angle_Discriminator, self).__init__()
#         self.device=device

#         self.encoder=Encoder(self.device)

#     def forward(self,x,y):
#         #Encoder 
#         f_x=self.encoder(x) #feature vector for x input
#         f_y=self.encoder(y) #feature vector for y input
#         return self.discriminator(f_x,f_y)

#     def discriminator(self,x,y):
#         """
#         Returns the angle between x and y feature vectors 
#         in degrees
#         """
#         x=x.view(x.shape[0],-1) # collapse 3D tensor to 2D tensor 
#         y=y.view(y.shape[0],-1) # collapse 3D tensor to 2D tensor
#         ndims=x.shape[1]        # get dimensionality of feature space
#         batch_size=x.shape[0]   # get batch_size
#         cos_angles=torch.zeros(batch_size,1).to(self.device)   
#         for i in range(0,ndims-1,2):
#             x_i=x[:,i:i+2]      
#             y_i=y[:,i:i+2]
#             dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
#             x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
#             y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)
#             cos_angles+=dot_prod/(x_norm*y_norm)
#         cos_angles=cos_angles/(ndims//2) # average
#         return (torch.acos(cos_angles)*180/np.pi) #turn into degrees



class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()

        self.device = device

        self.encoder=nn.Sequential(
            #1st Conv Layer
            nn.Conv2d(1,24,3),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #2nd Conv Layer
            nn.Conv2d(24,24,3),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #3rd Conv Layer
            nn.Conv2d(24,48,3,2),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #4th Conv Layer
            nn.Conv2d(48,48,3),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #5th Conv Layer
            nn.Conv2d(48,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #6th Conv Layer
            nn.Conv2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #7th Conv Layer
            nn.Conv2d(96,192,3),
            nn.BatchNorm2d(192),
            nn.RReLU(),
            #8th Conv Layer
            nn.Conv2d(192,192,3),
            )
    def forward(self,x):
         return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()

        self.device = device

        self.decoder=nn.Sequential(
            #1st dconv layer
            nn.ConvTranspose2d(192,192,3),
            nn.BatchNorm2d(192),
            nn.RReLU(),
            #2nd dconv layer
            nn.ConvTranspose2d(192,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #3rd dconv layer
            nn.ConvTranspose2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #4th dconv layer
            nn.ConvTranspose2d(96,48,3),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #5th dconv layer
            nn.ConvTranspose2d(48,48,3),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #6th dvonv layer
            nn.ConvTranspose2d(48,24,3,2,output_padding =1),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #7th dconv layer
            nn.ConvTranspose2d(24,24,3),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #8th dconv layer
            nn.ConvTranspose2d(24,1,3),
            nn.Sigmoid())

    def forward(self,x):
         return self.decoder(x)


