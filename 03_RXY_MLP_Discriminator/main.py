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
        self.down3 = nn.Linear(510, 255)
        self.down4=nn.Linear(255,3)

    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x= F.relu(self.down3(x))
        return F.sigmoid(self.down4(x))

def scale_tensor(input,max_scaling=2,plot=False):
    """Scale tesnosr
    Args:
        input:           [N,c,h,w] **numpy** tensor
        maximum_scaling: note that scaling should be symmetric in 
                         shrinking and magnification
        plot:            set flag about wether to print the transofromation or not 
    Returns:
        scaled output and scaling mapped into a [0,pi] scale
    """
    x_scale_pi=(np.pi)*np.random.rand(input.shape[0])
    y_scale_pi=(np.pi)*np.random.rand(input.shape[0])
  
    outputs =[]
    x_scale_real=np.zeros_like(x_scale_pi)
    y_scale_real=np.zeros_like(y_scale_pi)
    for i,_ in enumerate(input):
        #trasfrom scaling to real line
        if x_scale_pi[i]>=np.pi/2:
            x_scale=1+(x_scale_pi[i]-np.pi/2)/(np.pi/2)*(max_scaling-1)
        else:
            x_scale=1+(x_scale_pi[i]-np.pi/2)/(np.pi/2)/max_scaling

        if y_scale_pi[i]>=np.pi/2:
            y_scale=1+(y_scale_pi[i]-np.pi/2)/(np.pi/2)*(max_scaling-1)
        else:
            y_scale=1+(y_scale_pi[i]-np.pi/2)/(np.pi/2)/max_scaling

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


def save_model(model):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    epoch:  trainign epoch
    """
    import os
    if not os.path.exists('./model/'):
      os.mkdir('./model/')
    torch.save(model.state_dict(), './model/model.pt')


def se_loss(input,output):
    """
    Args:
        input/outout: [batch,3]  where 1st dim: x_scale [0,np.pi]
                                2nd dim: y_scale [0,np.pi]
                                3nd dmi: rotation [0,2*np.pi]


    """
    loss_x_scale=((output[:,0]-input[:,0]/np.pi)**2).mean()
    loss_y_scale=((output[:,1]-input[:,1]/np.pi)**2).mean()
    loss_rotation=((output[:,2]-input[:,2]/(2*np.pi))**2).mean()

    return loss_x_scale+loss_y_scale+loss_rotation

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
      
        params= torch.from_numpy(params).to(device)
        #data = data.view(data.size(0), -1)

        # Forward pass
        #data = data.to(device)
        optimizer.zero_grad()
        output = model(targets)

        #Loss
        loss=se_loss(params,output)

        # Backprop
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
            sys.stdout.flush()
    save_model(model,epoch)


def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
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
          
            params= torch.from_numpy(params).to(device)
            output = model(targets)

            loss_log=se_loss(params,output)
            break
    return loss_log


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
        save_images(output, args.epochs,'final',nrow=20)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
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
    parser.add_argument('--store-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before storing loss valies')

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

    train_loader_eval=torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Init model and optimizer

    training_loss=[]
    test_loss=[]
    path = "./output"

    model = Net(28,28,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
       # Where the magic happens
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
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
          
            params= torch.from_numpy(params).to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(targets)

            #Loss
            loss=se_loss(params,output)

            # Backprop
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                    .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
                sys.stdout.flush()

            if batch_idx % args.store_interval == 0:
                training_loss.append(test(args, model, device, train_loader_eval))
                test_loss.append(test(args, model, device, test_loader))
    #Save model
    save_model(model)
    #Save losses
    training_loss=np.array(training_loss)
    test_loss=np.array(test_loss)

    np.save(path+'/training_loss',training_loss)
    np.save(path+'/test_loss',test_loss)
    plot_learning_curve(args,training_loss,test_loss)

def plot_learning_curve(args,training_loss,test_loss):

    x_ticks=np.arange(len(training_loss))*args.store_interval*args.batch_size

    plt.plot(x_ticks,training_loss,label='Training Loss')
    plt.plot(x_ticks,test_loss,label='Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Training Examples')
    plt.title('Learning Curves')
    plt.legend()

    path = "./output/learning_curves"
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    # Create save path
    path = "./output"
    if not os.path.exists(path):
        os.makedirs(path)
    main()
