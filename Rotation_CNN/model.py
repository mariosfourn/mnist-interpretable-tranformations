import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

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


        # self.decoder=nn.Sequential(
        #     #1st dconv layer
        #     nn.Upsample(scale_factor=3), #3x3
        #     nn.Conv2d(192,192,3,padding=1), # 3x3
        #     nn.BatchNorm2d(192),
        #     nn.RReLU(),
        #     #2nd dconv layer
        #     nn.Upsample(scale_factor=2), # 6x6
        #     nn.Conv2d(192,96,3,padding=1), #6x6 
        #     nn.BatchNorm2d(96),
        #     nn.RReLU(),
        #     #3rd dconv layer
        #     nn.Upsample(scale_factor=2), #12x12
        #     nn.Conv2d(96,48,3,padding=1), #12x12 
        #     nn.BatchNorm2d(48),
        #     nn.RReLU(),
        #     #4th dconv layer
        #     nn.Upsample(scale_factor=2), #24x24
        #     nn.ConvTranspose2d(48,24,3,padding=2,stride=2), #14x14
        #     nn.BatchNorm2d(24),
        #     nn.RReLU(),
        #     #5th dconv layer
        #     nn.Upsample(scale_factor=2),# 28x28
        #     nn.ConvTranspose2d(24,1,3,padding=1), #28x28
        #     nn.Sigmoid())
        
            

    def forward(self, x, params):
        #Encoder 
        x=self.encoder(x)

        #Feature transform layer
        x=self.feature_transformer(x, params)

        #Deconder
        x=self.decoder(x)
          
        return x

    def feature_transformer(self, input, params):
        """For now we assume the params are just a single rotation angle

        Args:
            input: [N,c] tensor, where c = 2*int
            params: [N,1] tensor, with values in [0,2*pi)
        Returns:
            [N,c] tensor
        """
        # First reshape activations into [N,c/2,2,1] matrices
        x = input.view(input.size(0),input.size(1)/2,2,1)
        # Construct the transformation matrix
        sin = torch.sin(params)
        cos = torch.cos(params)
        transform = torch.cat([sin, -cos, cos, sin], 1)
        transform = transform.view(transform.size(0),1,2,2).to(self.device)
        # Multiply: broadcasting taken care of automatically
        # [N,1,2,2] @ [N,channels/2,2,1]
        output = torch.matmul(transform, x)
        # Reshape and return
        return output.view(input.size())


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