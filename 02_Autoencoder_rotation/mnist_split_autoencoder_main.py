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
import pandas as pd
import struct

import matplotlib
from scipy.ndimage.interpolation import rotate
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from model_v2 import Autoencoder_Split,Autoencoder_SplitMLP

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)


def rotate_tensor_give_angles(input,angles):
    """
    Rotates each image by angles and concatinates them in one tenosr
    Args:
        input: [N,c,h,w] **numpy** tensor
        angles: [D,1]
    Returns:
        output [N*D,c,h,w] **numpy** tensor
    """
    outputs = []
    for i in range(input.shape[0]):
        for angle in angles:
            output = rotate(input[i,...], 180*angle/np.pi, axes=(1,2), reshape=False)
            outputs.append(output)
    return np.stack(outputs, 0)


def rotate_tensor(input,init_rot_range,relative_rot_range, plot=False):
    """
    Rotate the image
    Args:
        input: [N,c,h,w] **numpy** tensor
        init_rot_range:     (scalar) the range of ground truth rotation
        relative_rot_range: (scalar) the range of relative rotations
        plot: (flag)         plot the original and rotated digits
    Returns:
        outputs1: [N,c,h,w]  input rotated by offset angle
        outputs2: [N,c,h,w]  input rotated by offset angle + relative angle [0, rot_range]
        relative angele [N,1] relative angle between outputs1 and outputs 2 in radians
    """
    #Define offest angle of input
    offset_angles=init_rot_range*np.random.rand(input.shape[0])
    offset_angles=offset_angles.astype(np.float32)

    #Define relative angle
    relative_angles=np.random.uniform(-relative_rot_range,relative_rot_range,input.shape[0])
    relative_angles=relative_angles.astype(np.float32)


    outputs1=[]
    outputs2=[]
    for i in range(input.shape[0]):
        output1 = rotate(input[i,...], offset_angles[i], axes=(1,2), reshape=False)
        output2 = rotate(input[i,...], (offset_angles[i]+relative_angles[i]), axes=(1,2), reshape=False)
        outputs1.append(output1)
        outputs2.append(output2)

    outputs1=np.stack(outputs1, 0)
    outputs2=np.stack(outputs2, 0)

    if plot:
        #Create of output1 and outputs1
        N=input.shape[0]
        rows=int(np.floor(N**0.5))
        cols=N//rows
        plt.figure()
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if outputs1.shape[1]>1:
                image=outputs1[j].transpose(1,2,0)
            else:
                image=outputs1[j,0]

            plt.imshow(image, cmap='gray')
            plt.grid(False)
            plt.title(r'$\theta$={:.1f}'.format(offset_angles[j]*180/np.pi), fontsize=6)
            plt.axis('off')
        #Create new figure with rotated
        plt.figure(figsize=(7,7))
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=outputs2[j].transpose(1,2,0)
            else:
                image=outputs2[j,0]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(r'$\theta$={:.1f}'.format( (offset_angles[i]+relative_angles[i])*180/np.pi), fontsize=6)
            plt.grid(False)
        plt.tight_layout()      
        plt.show()

    return outputs1, outputs2, relative_angles


def save_model(args,model,epoch):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    """
    path='./model_'+args.name
    if not os.path.exists(path):
      os.mkdir(path)
    model_name='checkpoint_epoch={}'.format(epoch)
    filepath=os.path.join(path,model_name)
    torch.save(model.state_dict(), filepath)


def reconstruction_test(args, model, test_loader, epoch,rot_range,path):

    model.eval()
    with torch.no_grad():
        for data,_ in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat

            data = data.view(data.size(0), -1)
            data = data.repeat(test_loader.batch_size,1)
            data = data.view(test_loader.batch_size**2,1, 28,28)
            targets= torch.zeros_like(data)

            angles = torch.linspace(0, rot_range, steps=test_loader.batch_size)
            angles = angles.view(test_loader.batch_size, 1)
            angles = angles.repeat(1, test_loader.batch_size)
            angles = angles.view(test_loader.batch_size**2, 1)


            # Forward pass
            output,_,_ =  model(data, targets,angles*np.pi/180) 
            break
        output = output.cpu()
        save_images(args, path, output, epoch)


def save_images(args,path,images, epoch, nrow=None):
    """Save the images in a grid format

    Args:
        images: array of shape [N,1,h,w], rgb=1 or 3
    """
    if nrow == None:
        nrow = int(np.floor(np.sqrt(images.size(0)
            )))

    img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True).numpy()
    img = np.transpose(img, (1,2,0))

    plt.figure()
    plt.imshow(img)
    plt.savefig(path+"/epoch{:04d}".format(epoch))
    plt.close()


def round_even(x):
    return int(round(x/2.)*2)



# class EucledianVectorLoss(nn.Module):
#     """
#     Penalty loss on feature vector to ensure that in encodes rotation information
#     """
    
#     def __init__(self, type , size_average=True):
#         super(EucledianVectorLoss,self).__init__()
#         self.size_average=size_average #flag for mena loss
#         self.type=type
        
#     def forward(self,x,y):
#         """
#         penalty loss bases on cosine similarity being 1

#         Args:
#             x: [batch,ndims]
#             y: [batch,ndims]
#         """
#         # x=x.view(x.shape[0],-1)
#         # y=y.view(y.shape[0],-1)
#         #Number of features
#         ndims=x.shape[1]
#         #Batch size


#         reg_loss=0.0

#         cosine_similarity=nn.CosineSimilarity(dim=2)

#         for i in range(0,ndims-1,2):
#             x_i=x[:,i:i+2]
#             y_i=y[:,i:i+2]
#             # dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
#             # x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
#             # y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

#             if self.type=='mse':
#                 reg_loss+= ((cosine_similarity(x_i.view(x_i.size(0),1,2),y_i.view(y_i.size(0),1,2))-1.0)**2).sum()
#                 #reg_loss+=((dot_prod/(x_norm*y_norm)-1)**2).sum()
#             elif self.type=='abs':
               
#                 reg_loss+= torch.abs(cosine_similarity(x_i.view(x_i.size(0),1,2),y_i.view(y_i.size(0),1,2))-1.0).sum()
#                 #eg_loss+=(abs(dot_prod/(x_norm*y_norm)-1)).sum()
              
#             elif self.type=='L2_norm':
#                 forb_distance=torch.nn.PairwiseDistance()
#                 x_polar=x_i/torch.maximum(x_norm, 1e-08)
#                 y_polar=y_i/torch.maximum(y_norm,1e-08)
#                 reg_loss+=(forb_distance(x_polar,y_polar)**2).sum()
           
#         if self.size_average:
#             reg_loss=reg_loss/x.shape[0]/(ndims//2)

#         return reg_loss



def triple_loss(args,targets, output,identity, eucledian):
    """
    output: [N,1,28,28] tensor
    identity: tuple of 2x [N,192-args.num_dims,1,1 ] tensors
    eucledian: tuple of 2x [N,args.num_dims,1,1 ] tensors


    """
    #Extract arguments

    x_eucledian=eucledian[0] # [N,args.num_dims,1,1 ]

    x_eucledian=x_eucledian.view(-1,x_eucledian.size(1)) #collapse to 2d  [N,args.num_dims]

    y_eucledian=eucledian[1]  #[N,args.num_dims,1,1 ]

    y_eucledian=y_eucledian.view(-1,y_eucledian.size(1)) #collapse to 2d  [N,args.num_dims]

    x_identity=identity[0] #[N,512-args.num_dims,1,1 ]

    x_identity=x_identity.view(-1,x_identity.size(1)) #[N,512-args.num_dims]

    y_identity=identity[1] #[N,512-args.num_dims,1,1 ]

    y_identity=y_identity.view(-1,y_identity.size(1)) #[N,512-args.num_dims]

    #1 Reconstriction Loss

    reconstruction_loss=nn.BCELoss(reduction='elementwise_mean')
    recon_loss=reconstruction_loss(output,targets)

    #2 Eucledian space loss
    cosine_similarity=nn.CosineSimilarity(dim=2)

    rotation_loss=torch.abs(cosine_similarity(x_eucledian.view(x_eucledian.size(0),1,2),
        y_eucledian.view(y_eucledian.size(0),1,2))-1.0).mean()

    # rotation_loss=eucledian_loss(x_eucledian,y_eucledian)

    #3 Idenity loss (L2 distance)

    identity_loss=F.pairwise_distance(x_identity/x_identity.norm(dim=1,keepdim=True)
        ,y_identity/y_identity.norm(dim=1,keepdim=True), p=2)
    
    identity_loss=identity_loss.mean()
 
    total_loss=args.alpha*rotation_loss + args.gamma * identity_loss + (1-args.alpha-args.gamma)*recon_loss


    return (total_loss,rotation_loss,identity_loss,recon_loss)


def evaluate_model(args,model,data_loader):
    """
    Evaluate loss for input data loader
    """
    model.eval()
    with torch.no_grad():
        for data,_ in data_loader:

            data,targets,angles = rotate_tensor(data.numpy(),args.init_rot_range, args.train_rotation_range)
            data = torch.from_numpy(data)
            targets = torch.from_numpy(targets)
            angles = torch.from_numpy(angles)
            angles = angles.view(angles.size(0), 1)

            # Forward pass
            output, identity_vectors, eucleidian_vectors= model(data, targets,angles*np.pi/180) 
           # Get triplet loss
            losses=triple_loss(args,targets,output, identity_vectors, eucleidian_vectors)
            break

    return losses[0].item()
 

def main():

    # Training settings
    list_of_choices=['mse','abs','L2_norm']
    list_of_models=['normal', 'MLP']
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size-recon', type=int, default=10, metavar='N',
                        help='input batch size for reconstruction testing (default: 10)')
    parser.add_argument('--test-batch-size-rot', type=int, default=1000, metavar='N',
                        help='input batch size for rotation disrcimination testing (default: 1,000)')
    parser.add_argument('--batch-size-eval', type=int, default=100, metavar='N',
                        help='batch size for evaluation of error on MNSIT digits (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--alpha',type=float, default=0.5, metavar='a',
                        help='proportion of rotation loss')
    parser.add_argument('--gamma',type=float, default=0.0, metavar='g',
                        help='proportion of identity loss')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument("--loss",dest='loss',default='abs',
    choices=list_of_choices, help='Decide type of penatly loss [mse or abs] (Default=abs)') 
    parser.add_argument('--step',type=int, default=5,
                        help='Size of step in degrees for evaluation of error at end of traning (Default=5)')
    parser.add_argument('--num-dims',type=int, default=2,
                        help='Number of feature vector dimension to use for rotation estimation (Default=2)')
    parser.add_argument('--init-rot-range',type=float, default=0,
                        help='Upper bound of range in degrees of initial random rotation of digits, (Default=0)')
    parser.add_argument('--train-rotation-range', type=float, default=90, metavar='theta',
                        help='rotation range in degrees for training,(Default=180), [-theta,+theta)')
    parser.add_argument('--eval-rotation-range', type=float, default=90, metavar='theta',
                        help='rotation range in degrees for evaluation,(Default=90), [-theta,+theta)')
    parser.add_argument("--model-type",default='normal',
    choices=list_of_models, help='model type (Default=normal)') 
    parser.add_argument('--lr-scheduler', action='store_true', default=False, 
                        help='set up lernaring rate scheduler (Default off)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait until learning rate is reduced in plateua (default=3)')
    parser.add_argument('--threshold', type=float, default=1e-4, metavar='l',
                        help='ReduceLROnPlateau signifance threshold (Default=1e-4)')


    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(args.seed)

    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg,  getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format( torch.initial_seed()))
    sys.stdout.flush()

    logging_dir='./logs_'+args.name

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    torch.manual_seed(args.seed)

    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True)

    train_loader_recon = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size_recon, shuffle=True)

    train_loader_rotation = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size_rot, shuffle=True)

    # Init model and optimizer


    if args.model_type=='normal':
        model = Autoencoder_Split(args.num_dims)
    else:
        model = Autoencoder_SplitMLP()

    writer = SummaryWriter(logging_dir, comment='Split Autoencoder for MNIST')
  
    #Initialise weights
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            patience=args.patience,verbose=True,threshold=args.threshold)


    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    test_error_mean_log=[]
    test_error_std_log=[]
    n_iter=0

    for epoch in range(1, args.epochs + 1):

        sys.stdout.write('Epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data, _) in enumerate(train_loader):
            model.train()
            # Reshape data
            data,targets,angles = rotate_tensor(data.numpy(),args.init_rot_range, args.train_rotation_range)
            data = torch.from_numpy(data)
            targets = torch.from_numpy(targets)
            angles = torch.from_numpy(angles)
            angles = angles.view(angles.size(0), 1)

            # Forward pass
            output, identity_vectors, eucleidian_vectors= model(data, targets,angles*np.pi/180) 
            optimizer.zero_grad()
            #(output of autoencoder, feature vector of input, feature vector of rotated data)
            output, f_data, f_targets = model(data, targets,angles) 

           # Get triplet loss
            losses=triple_loss(args,targets,output, identity_vectors, eucleidian_vectors)
            # Backprop
            losses[0].backward()
            optimizer.step()       

            #Log progress
            if batch_idx % args.log_interval == 0:
                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                   .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), losses[0].item()))
                sys.stdout.flush()

                writer.add_scalars('Mini-batch loss',{'Total Loss':  losses[0].item(),
                                       'Rotation Loss ': losses[1].item(),
                                       'Identity Loss': losses[2].item(),
                                      ' Reconstruction Loss ': losses[-1].item() }, n_iter)


            #Store training and test loss
            if batch_idx % args.store_interval==0:
                error_mean, error_std = eval_synthetic_rot_loss(args, model,train_loader_rotation)
                writer.add_scalars('Rotation Loss',{'Mean Error': error_mean, 'Error STD:error_std':error_std}, n_iter)

            n_iter+=1



        #Evalaute training loss
        train_loss=evaluate_model(args,model,train_loader_rotation)
        writer.add_scalar('Training Loss',train_loss,epoch)
        if args.lr_scheduler: scheduler.step(train_loss)



            #     #Evaluate loss in 1,0000 sample of the traning set
        
            #     recon_loss, pen_loss=evaluate_model(args,device,model,train_loader_rotation)
            #     recon_train_loss.append(recon_loss.item())
            #     penalty_train_loss.append(pen_loss.item())

            #     # Average prediction error in degrees
            #     average, std=rotation_test(args, model, device,train_loader_rotation)
            #     prediction_avarege_error.append(average)
            #     prediction_error_std.append(std)
        if epoch % 5==0:
            #Test reconstruction by printing image
            reconstruction_test(args, model,train_loader_recon, epoch, args.eval_rotation_range,logging_dir)
            save_model(args,model,epoch)


    #Save model
    

def convert_to_convetion(input):
    """
    Coverts all anlges to convecntion used by atan2
    """

    input[input<180]=input[input<180]+360
    input[input>180]=input[input>180]-360
    
    return input


def eval_synthetic_rot_loss(args,model,data_loader):

    model.eval()
    #Number of features penalised
    
    with torch.no_grad():
        for data,_ in data_loader:
            ## Reshape data
            data,targets,angles = rotate_tensor(data.numpy(),args.init_rot_range, args.eval_rotation_range)
            data=torch.from_numpy(data)
            targets=torch.from_numpy(targets)

            # Forward passes
            f_data=model.encoder(data)
            if isinstance(f_data, tuple): f_data=f_data[1]
            f_data=f_data.view(f_data.shape[0],-1) #convert 3D vector to 2D

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinate 

            f_targets=model.encoder(targets)
            if isinstance(f_targets, tuple): f_targets=f_targets[1]
            f_targets=f_targets.view(f_targets.shape[0],-1) #convert 3D vector to 2D

            f_targets_y= f_targets[:,1] #Extract y coordinates
            f_targets_x= f_targets[:,0] #Extract x coordinate 

            theta_data=torch.atan2(f_data_y,f_data_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector

            estimated_angle=theta_targets-theta_data
            
            estimated_angle=convert_to_convetion(estimated_angle)

            error=estimated_angle-angles

            break
           
    return  abs(error).mean(), error.std()

def plot_learning_curve(args,recon_loss,penatly_loss,average_error,error_std,path):

    x_ticks=np.arange(len(recon_loss))*args.store_interval*args.batch_size
    with plt.style.context('ggplot'):

        fig, (ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(5,5))
        total_loss=recon_loss+args.Lambda*penatly_loss
       
        #Plot loss
        ax1.plot(x_ticks,recon_loss,label='Reconstruction (BCE) training loss',linewidth=1.25)
        ax1.plot(x_ticks,penatly_loss,label='Penalty training loss',linewidth=1.25)
        ax1.plot(x_ticks,total_loss,label ='Total training Loss',linewidth=1.25)
        ax1.set_ylabel('Loss',fontsize=10)
        
        ax1.legend()

        line,=ax2.plot(x_ticks,average_error,label='Average Abs training error',linewidth=1.25,color='g')
        ax2.fill_between(x_ticks,average_error-error_std,average_error+error_std,
            alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        ax2.set_ylabel('Degrees',fontsize=10)
        ax2.set_xlabel('Training Examples',fontsize=10)
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax2.legend()


        #Control colour of ticks
        ax1.tick_params(colors='gray', direction='out')
        for tick in ax1.get_xticklabels():
            tick.set_color('gray')
        for tick in ax1.get_yticklabels():
            tick.set_color('gray')

        ax2.tick_params(colors='gray', direction='out')
        for tick in ax2.get_xticklabels():
            tick.set_color('gray')
        for tick in ax2.get_yticklabels():
            tick.set_color('gray')

        fig.suptitle(r'Learning Curves $\lambda$={}'.format(args.Lambda))
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        fig.savefig(path+'/learning_curves')
        fig.clf()

if __name__ == '__main__':
    main()
