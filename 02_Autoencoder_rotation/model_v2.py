import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


def feature_transformer(input, params):
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
    transform = transform.view(transform.size(0),1,2,2)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


def inverse_feature_transformer(input, params):
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
    transform = transform.view(transform.size(0),1,2,2)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


class Autoencoder_Split(nn.Module):
    """Autoencoder with regularisatoon """
    def __init__(self,num_dims):
        super(Autoencoder_Split, self).__init__()

        self.num_dims=num_dims

        self.encoder=Encoder()
        self.decoder=Decoder(192)

    def forward(self,x,y,params):
        """
        Args:
            x:      untransformed images pytorch tensor
            y:      transforedm images  pytorch tensor
            params: rotations
        """
        f_x=self.encoder(x) #feature vector for image x [N,192,1,1]

        f_y=self.encoder(y) #feature vector for image y [N,192,1,1]

        #Apply FTL only on eucledian vector of f_x

        f_x[:,:self.num_dims]=feature_transformer(f_x[:,:self.num_dims], params)
        
        #Decoder
        output=self.decoder(f_x)

        #Split vector to orientation and identity 
        Eucledian_Vector_x=f_x[:,:self.num_dims]

        Identity_Vector_x=f_x[:,self.num_dims:]

        #Images y

        Eucledian_Vector_y=f_y[:,:self.num_dims]

        Identity_Vector_y=f_y[:,self.num_dims:]

        #Apply FTL on x

        #Return reconstructed image, feature vector of oringial image, feature vector of transformation
        return output, (Identity_Vector_x,Identity_Vector_y),(Eucledian_Vector_x,Eucledian_Vector_y)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    

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
    def __init__(self,num_dims):
        super(Decoder, self).__init__()

        self.decoder=nn.Sequential(
            #1st dconv layer
            nn.ConvTranspose2d(num_dims,num_dims,3),
            nn.BatchNorm2d(num_dims),
            nn.RReLU(),
            #2nd dconv layer
            nn.ConvTranspose2d(num_dims,96,3),
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


