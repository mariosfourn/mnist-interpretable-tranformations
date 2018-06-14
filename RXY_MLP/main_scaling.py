from __future__ import print_function
import os
import sys
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from itertools import cycle

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms
import cv2


# This function rounds to closest even number
def round_even(x):
    return int(round(x/2.)*2)

class Net(nn.Module):
    def __init__(self, height, width, device):
        super(Net, self).__init__()

        self.height = height
        self.width = width
        self.device = device

        # Init model layers
        self.down1 = nn.Linear(self.height*self.width, 510)
        self.down2 = nn.Linear(510, 510)
        self.down3 = nn.Linear(510, 510)
        self.up3 = nn.Linear(510, 510)
        self.up2 = nn.Linear(510, 510)
        self.up1 = nn.Linear(510, self.height*self.width)


    def forward(self, x, params):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.down3(x)   # Must be linear layer!

        # Feature transform layer
        x = self.feature_transformer(x, params)

        x = F.relu(self.up3(x))
        x = F.relu(self.up2(x))
        return F.sigmoid(self.up1(x))   # Sigmoid output for MNIST


    def feature_transformer(self, input, params):
        """feature transform layer

        Args:
            input: [N,c] tensor, where c = 6*int
            params: [N,3] tensor, with values in [0,2*pi)
        Returns:
            [N,c] tensor
        """
        # First reshape activations into [N,c/6,3,2,1] matrices
        x = input.view(input.size(0),input.size(1)/6,3,2,1)
        # Construct the transformatio  matrix
        params=params.unsqueeze(-1)
        sin = torch.sin(params)
        cos = torch.cos(params)
        transform = torch.cat([sin, -cos, cos, sin], -1) #[N,3,4]
        #Reshape to allow broadcasting [N,1,3,2,2]
        
        transform = transform.view(transform.size(0),1,transform.size(1),2,2).to(self.device) 
        # Multiply: broadcasting taken care of automatically
        # [N,1,3,2,2] @ [N,c/6,3,2,1]

        output = torch.matmul(transform, x) # [N,c/6,3,2,1]

        # Reshape and return
        return output.view(input.size())


def scale_tensor(input,max_scaling=2,plot=False):
    """Scale tesnosr
    Args:
        input:           [N,c,h,w] **numpy** tensor
        maximum_scaling: note that scaling should be symmetric in 
                         shrinking and magnification
        plot:            set flag about wether to print the transofromation or not 
    Returns:
        scaled output and scaling mapped into a [-pi,pi] scale
    """
    x_scale_pi=(np.pi/2)*np.random.uniform(-1,1,input.shape[0])
    y_scale_pi=(np.pi/2)*np.random.uniform(-1,1,input.shape[0])
  
    outputs =[]
    x_scale_real=np.zeros_like(x_scale_pi)
    y_scale_real=np.zeros_like(y_scale_pi)
    for i,_ in enumerate(input):
        #trasfrom scaling to real line
        if x_scale_pi[i]>=0:
            x_scale=1+x_scale_pi[i]/(np.pi/2)*(max_scaling-1)
        else:
            x_scale=1+x_scale_pi[i]/(np.pi/2)/max_scaling

        if y_scale_pi[i]>=0:
            y_scale=1+y_scale_pi[i]/(np.pi/2)*(max_scaling-1)
        else:
            y_scale=1+y_scale_pi[i]/(np.pi/2)/max_scaling

        x_scale_real[i]=x_scale
        y_scale_real[i]=y_scale
        new_size=[round_even(28*y_scale),round_even(28*x_scale)]
        assert (new_size[0]>=14  and new_size[1]>=14), (x_scale,x_scale_pi[i], y_scale,y_scale_pi[i])
        # Resize image 
        #tranpose input image to [h,w,c]
        channels=input.shape[1]
        image=np.transpose(input[i],(1,2,0))
        resized_image=cv2.resize(image,tuple(new_size[::-1]), interpolation = cv2.INTER_AREA)
        #Expand axis if the image is single channel
        if len(resized_image.shape)<3: resized_image= np.expand_dims(resized_image, axis=2)
        #Pad with zeros
        pos_diff = np.maximum(-np.asarray(resized_image.shape[:2]) + np.asarray([28,28]), 0)
        paddings = ((pos_diff[0]//2, pos_diff[0] - (pos_diff[0]//2)),
            (pos_diff[1]//2, pos_diff[1] - (pos_diff[1]//2)),(0,0))
        padded_image = np.pad(resized_image, paddings,'constant')
        # Now to crop
        crop_diff = np.asarray(padded_image.shape[:2]) - np.asarray([28,28])
        left_crop = crop_diff // 2
        right_crop = crop_diff - left_crop
        right_crop[right_crop==0] = -28
        new_image = padded_image[left_crop[0]:-right_crop[0], left_crop[1]:-right_crop[1],:]

        assert new_image.shape==(28,28,channels), new_image.shape
        outputs.append(new_image.transpose(2,0,1)) #[c,h,w]
    outputs=np.stack(outputs, 0) #[N,c,h,w]

    if plot:
        import matplotlib.pyplot as plt
        #Create a grid plot with original and scaled images
        N=input.shape[0]
        rows=int(np.floor(N**0.5))
        cols=N//rows

        #Create figure with original data
        plt.figure()
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=input[j].transpose(1,2,0)
            else:
                image=input[j,0]

            plt.imshow(image, cmap='gray')
            plt.grid(False)
            plt.axis('off')
        #Create new figure with scaled data
        plt.figure(figsize=(7,7))
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=outputs[j].transpose(1,2,0)
            else:
                image=outputs[j,0]
            plt.imshow(image, cmap='gray')
            plt.xlabel('x: {:.2f} '.format(x_scale_real[j]),fontsize=6)
            plt.ylabel('y: {:.2f}'.format(y_scale_real[j]),fontsize=6)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()      
        plt.show()
    return outputs, x_scale_pi, y_scale_pi


def rotate_tensor(input,plot=False):
    """Nasty hack to rotate images in a minibatch, this should be parallelized
    and set in PyTorch

    Args:
        input: [N,c,h,w] **numpy** tensor
    Returns:
        rotated output and angles in radians
    """
    angles = 2*np.pi*np.random.rand(input.shape[0])
    angles = angles.astype(np.float32)
    outputs = []
    for i in range(input.shape[0]):
        output = rotate(input[i,...], 180*angles[i]/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)

    outputs=np.stack(outputs, 0)

    if plot:
        import matplotlib.pyplot as plt
        #Create a grid plot with original and scaled images
        N=input.shape[0]
        rows=int(np.floor(N**0.5))
        cols=N//rows
        plt.figure()
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=input[j].transpose(1,2,0)
            else:
                image=input[j,0]

            plt.imshow(image, cmap='gray')
            plt.grid(False)
            plt.axis('off')
        #Create new figure with rotated
        plt.figure(figsize=(7,7))
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=outputs[j].transpose(1,2,0)
            else:
                image=outputs[j,0]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(r'$\theta$={:.1f}'.format( angles[j]*180/np.pi), fontsize=6)
            plt.grid(False)
        plt.tight_layout()      
        plt.show()

    return outputs, angles


def save_model(model,epoch):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    epoch:  trainign epoch
    """
    import os
    if not os.path.exists('./model/'):
      os.mkdir('./model/')
    torch.save(model.state_dict(), './model/ckpoint_epoch_{}'.format(epoch))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data
        targets,x_scale,y_scale= scale_tensor(data.numpy())
        targets, angles = rotate_tensor(targets)
       
        targets = torch.from_numpy(targets).to(device)
        targets = targets.view(targets.size(0), -1)

        #Create parameters 2D tensor
        x_scale=x_scale.reshape(-1,1).astype(np.float32)
        y_scale=y_scale.reshape(-1,1).astype(np.float32)
        angles=angles.reshape(-1,1).astype(np.float32)
        params= np.hstack((x_scale, y_scale, angles))
        #params= create_params(angles,x_scale,y_scale)
        params= torch.from_numpy(params).to(device)
        #params = params.view(params.size(0), 3)
        data = data.view(data.size(0), -1)

        # Forward pass
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, params)

        # Binary cross entropy loss
        loss_fnc = nn.BCELoss(size_average=False)
        loss = loss_fnc(output, targets)

        # Backprop
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()
    save_model(model,epoch)


def test(args, model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            data = data.view(data.size(0), -1)
            data = data.repeat(args.test_batch_size,1)

            angles = torch.linspace(0, 2*np.pi, steps=args.test_batch_size).unsqueeze(1)
            scale= torch.linspace(-np.pi/2, np.pi/2, steps=args.test_batch_size).unsqueeze(1)
            zeros=torch.zeros_like(scale)
            #Create parameter vector only with angles
            params_angles=torch.cat((zeros,zeros,angles),1).repeat(1,args.test_batch_size
                ).view(args.test_batch_size**2,3)
            
            params_scale=torch.cat((scale,scale,zeros),1).repeat(1,args.test_batch_size
                ).view(args.test_batch_size**2,3)

            # Forward pass
            data = data.to(device)
            output_angles= model(data, params_angles)
            output_scale=model(data, params_scale)
            break
        output_angles = output_angles.cpu()
        output_angles = output_angles.view(-1,1,28,28)
        output_scale = output_scale.cpu()
        output_scale = output_scale.view(-1,1,28,28)

        save_images(output_angles, epoch,'angles')
        save_images(output_scale,epoch,'scaleXY')


def save_images(images, epoch, flag=None,nrow=None):
    """Save the images in a grid format

    Args:
        images: array of shape [N,1,h,w], rgb=1 or 3
    """
    if nrow == None:
        nrow = int(np.floor(np.sqrt(images.size(0))))
    if flag== None: flag=''

    img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True).numpy()
    img = np.transpose(img, (1,2,0))

    plt.figure()
    plt.grid(False)
    plt.imshow(img)
    plt.savefig("./output/epoch{:04d}".format(epoch)+flag)
    plt.close()

def final_test(args,model,device,test_loader):
    #parameter vector only of x_scaling 
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            steps=20
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            data = data.view(data.size(0), -1)
            data = data[:7,:]
            data = data.repeat(1,steps).view(data.size(0)*steps,-1)
            scale= torch.linspace(-np.pi/2, np.pi/2, steps=steps).unsqueeze(1)
            angles = torch.linspace(0, 2*np.pi, steps=steps).unsqueeze(1)

            #params for only x-scaling
            zeros=torch.zeros_like(scale)
            params_x=torch.cat((scale,zeros,zeros),1)
            params_y=torch.cat((zeros,scale,zeros),1)
            params_r=torch.cat((zeros,zeros,angles),1)
            params_xy=torch.cat((scale,scale,zeros),1)
            params_rx=torch.cat((scale,zeros,angles),1)
            params_ry=torch.cat((zeros,scale,angles),1)
            params_rxy=torch.cat((scale,scale,angles),1)
            params=torch.cat((params_x,params_y,params_r,params_xy,params_rx,params_ry,params_rxy),0)

            # Forward pass
            data = data.to(device)
            output= model(data, params)
            break
        output = output.cpu()
        output= output.view(-1,1,28,28)

        #Plot images
        save_images(output, args.epochs,'final',nrow=7)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Init model and optimizer
    model = Net(28,28,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Where the magic happens
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)
    final_test(args,model,device,test_loader)


if __name__ == '__main__':
    # Create save path
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    main()
