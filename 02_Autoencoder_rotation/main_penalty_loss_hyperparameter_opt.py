from __future__ import print_function
import os
import sys
import itertools
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

from model import Net_Reg

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


def rotate_tensor(input,rot_range=np.pi,plot=False):
    """
    Rotate input tensor in range [0, rot_range] randomnly

    Args:
        input: [N,c,h,w] **numpy** tensor
    Returns:
        rotated output and angles in radians
    """
    angles = rot_range*np.random.rand(input.shape[0])
    angles = angles.astype(np.float32)
    outputs = []
    for i in range(input.shape[0]):
        output = rotate(input[i,...], 180*angles[i]/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)

    outputs=np.stack(outputs, 0)
    if plot:
        #Create a grid plot with original and scaled images
        N=input.shape[0]
        rows=int(np.floor(N**0.5))
        cols=N//rows

        plt.figure()

        for j in range(rows*cols):
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
        for j in range(rows*cols):
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


def save_model(args,model):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    """
    path='./model_'+args.name
    import os
    if not os.path.exists(path):
      os.mkdir(path)
    torch.save(model.state_dict(), path+'/checkpoint.pt')


def reconstruction_test(args, model, device, test_loader, epoch,rot_range=np.pi):

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            data = data.view(data.size(0), -1)
            data = data.repeat(test_loader.batch_size,1)
            data = data.view(test_loader.batch_size**2,1, 28,28)
            target = torch.zeros_like(data)

            angles = torch.linspace(0, rot_range, steps=test_loader.batch_size)
            angles = angles.view(test_loader.batch_size, 1)
            angles = angles.repeat(1, test_loader.batch_size)
            angles = angles.view(test_loader.batch_size**2, 1)


            # Forward pass
            data = data.to(device)
            target=target.to(device)
            output,_,_ = model(data, target, angles)
            break
        output = output.cpu()
        save_images(args,output, epoch)


def save_images(args,images, epoch, nrow=None):
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
    path = "./output_" +args.name
    plt.savefig(path+"/epoch{:04d}".format(epoch))
    plt.close()


def round_even(x):
    return int(round(x/2.)*2)

class Penalty_Loss(nn.Module):
    """
    Penalty loss on feature vector to ensure that in encodes rotation information
    """
    
    def __init__(self,proportion=1.0, size_average=False,type='mse'):
        super(Penalty_Loss,self).__init__()
        self.size_average=size_average #flag for mena loss
        self.proportion=proportion     #proportion of feature vector to be penalised
        self.type=type
        
    def forward(self,x,y):
        """
        penalty loss bases on cosine similarity being 1

        Args:
            x: [batch,1,ndims]
            y: [batch,1,ndims]
        """
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        #Number of features
        total_dims=x.shape[1]
        #Batch size
        batch_size=x.shape[0]

        #Number of features penalised
        ndims=round_even(self.proportion*total_dims)
        reg_loss=0.0

        for i in range(0,ndims-1,2):
            x_i=x[:,i:i+2]
            y_i=y[:,i:i+2]
            dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
            x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
            y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

            if type=='mse':
                reg_loss+=((dot_prod/(x_norm*y_norm)-1)**2).sum()
            else:
                reg_loss+=(abs(dot_prod/(x_norm*y_norm)-1)).sum()
                
        if self.size_average:
            reg_loss=reg_loss/x.shape[0]/(ndims//2)
        return reg_loss


def penalised_loss(args,output,targets,f_data,f_targets,prop, Lambda):
    """
    Define penalised loss
    """

    # Binary cross entropy loss
    loss_fnc = nn.BCELoss(size_average=True)
    loss_reg = Penalty_Loss(size_average=True,proportion=prop,type=args.loss)
    #Add 
    reconstruction_loss=loss_fnc(output,targets)
    rotation_loss=loss_reg(f_data,f_targets)
   
    total_loss= reconstruction_loss+Lambda*rotation_loss
    return total_loss,reconstruction_loss,rotation_loss


def evaluate_model(args,device,model,data_loader):
    """
    Evaluate loss for input data loader
    """
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            # Reshape data
            targets, angles = rotate_tensor(data.numpy())
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)

            # Forward pass
            data = data.to(device)
           
            output, f_data, f_targets = model(data, targets,angles) #for feature vector
            loss,reconstruction_loss,penalty_loss=penalised_loss(args,output,targets,f_data,f_targets)
            break

    return reconstruction_loss,penalty_loss
 


def read_idx(filename):
    import struct
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def get_metrics(model, data_loader,device, step,prop):
    """ 
    Returns the average error per step of degrees in the range [0,np.pi]
    Args:
        model : pytorch Net_Reg model
        data_loader
        step (scalar) in degrees
    
    """
    #turn step to radians
    step=np.pi*step/180
    entries=int(np.pi/step)
    model.eval()
    errors=np.zeros((entries,len(data_loader.dataset)))
    
    
    with torch.no_grad():

        start_index=0
        for batch_idx,data in enumerate(data_loader):
            
            batch_size=data.shape[0]
            angles = torch.arange(0, np.pi, step=step)
            target = rotate_tensor_give_angles(data.numpy(),angles.numpy())
            data=data.to(device)

            
            
            target = torch.from_numpy(target).to(device)
            
            #Get Feature vector for original and tranformed image

            x=model.encoder(data) #Feature vector of data
            y=model.encoder(target) #Feature vector of targets

            #Compare Angles            
            x=x.view(x.shape[0],1,-1)
            x=x.repeat(1,entries,1)# Repeat each vector "entries" times
            x=x.view(-1,x.shape[-1])# collapse 3D tensor to 2D tensor
            
            y=y.view(y.shape[0],-1) # collapse 3D tensor to 2D tensor
            
            new_batch_size=x.shape[0]   # get augmented batch_size
            
            #Loop every 2 dimensions

             #Number of features
            total_dims=x.shape[1]
            #Number of features penalised
            ndims=round_even(prop*total_dims)  
            
            sys.stdout.write("\r%d%% complete \n" % ((batch_idx * 100)/len(data_loader)))
            sys.stdout.flush()
            angles_estimate=torch.zeros(new_batch_size,1).to(device)  
            
       
            for i in range(0,ndims-1,2):
                x_i=x[:,i:i+2]      
                y_i=y[:,i:i+2]
                
                #Get dot product for the batch
                dot_prod=torch.bmm(x_i.view(new_batch_size,1,2),y_i.view(new_batch_size,2,1)).view(new_batch_size,1)

                #Get euclidean norm
                x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
                y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

                #Get the cosine of the angel for example
                angles_estimate+=dot_prod/(x_norm*y_norm)
            
            
            angles_estimate=torch.acos(angles_estimate/(ndims//2))*180/np.pi # average and in degrees
            angles_estimate=angles_estimate.cpu()
            error=angles_estimate.numpy()-(angles.view(-1,1).repeat(batch_size,1).numpy()*180/np.pi)
            
           
            
            #Get the tota
            for i in range(entries):
                index=np.arange(i,new_batch_size,step=entries)
                errors[i,start_index:start_index+batch_size]=error[index].reshape(-1,)

            start_index+=batch_size
    
    mean_error=errors.mean(axis=1)
    mean_abs_error=(abs(errors)).mean(axis=1)
    error_std=errors.std(axis=1, ddof=1)
   
    return mean_error, mean_abs_error, error_std


class MNISTDadataset(Dataset):
    def __init__(self,root_dir, digit,transform=None):
        """
        Args:
            digit(int):        MNIST digit
            root_dir (string): Directory where the ubyte lies
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        file_path =os.path.join(root_dir,'train-images-'+str(digit)+'-ubyte')
        self.data = read_idx(file_path)/255
        
        self.data = torch.Tensor(self.data)
        self.data =  (self.data).unsqueeze(1)
        
        self.transform=transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample=self.data[idx]
        

        if self.transform:
            sample = self.transform(sample)

        return sample


def  get_error_per_digit(args,model,batch_size, step,digit,prop):
    """
    Returns the mean absolute mean error and std for the digit on MNIST
    Args:
        step (scalar):  rotation step in degrees
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #Load Dataset for each MNIST digit
    data_loader=DataLoader(MNISTDadataset('./data/',digit), 
        batch_size=batch_size, shuffle=False, **kwargs)

    sys.stdout.write('Processing digit {} \n'.format(digit))
    sys.stdout.flush()
    results=get_metrics(model, data_loader,device, step,prop)
    mean_abs_error=results[1]
    error_std=results[2]

    return mean_abs_error,error_std,

def main():

    # Training settings
    list_of_choices=['mse','abs']
    parser = argparse.ArgumentParser(description='Hyper-Parametere optimisation')
    parser.add_argument('--digit', type=int, default=5,
                        help='digit to be used for evaluation metrics (default=5') 
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument("--loss",dest='loss',default='mse',
    choices=list_of_choices, help='Decide type of penatly loss, mse (defautl) or abs')  
    parser.add_argument('--step',type=int, default=5,
                        help='Size of step in degrees for evaluation of error at end of traning')
    args = parser.parse_args()

    # Create save path
    path = "./output_" +args.name
    if not os.path.exists(path):
        os.makedirs(path)

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

    train_loader_recon = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size_recon, shuffle=True, **kwargs)

    train_loader_rotation = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size_rot, shuffle=True, **{})

    #Create empty pandad dataframe to popopylate with mean adn std of error

    #Iteratte

    average_abs_error=pd.DataFrame()
    error_std=pd.DataFrame()
    prop_list=[1.0, 0.6, 0.4, 0.2, 0.1]
    Lambda_list=[1.0, 2.0, 3.0, 4.0, 5.0]
    combinations=[]
    num_iter=len(list(itertools.product(prop_list,Lambda_list)))
    for iter,(prop, Lambda) in enumerate(itertools.product(prop_list,Lambda_list)):
        sys.stdout.write('Start training model {}/{}\n'.format(iter+1,num_iter))
        sys.stdout.flush()

        # Init model and optimizer
        #prop, Lambda= get_sample()
        combinations.append([prop, Lambda])

        model_name='pror_{:.2f}_Lambda_{:.2f}'.format(prop, Lambda)

        model = Net_Reg(device).to(device)
      
        #Initialise weights
        model.apply(weights_init)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)



        # Where the magic happens
        for epoch in range(1, args.epochs + 1):
            sys.stdout.write('Epoch {}/{} \n '.format(epoch,args.epochs))
            sys.stdout.flush()
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                # Reshape data
                targets, angles = rotate_tensor(data.numpy())
                targets = torch.from_numpy(targets).to(device)
                angles = torch.from_numpy(angles).to(device)
                angles = angles.view(angles.size(0), 1)

                # Forward pass
                data = data.to(device)
                optimizer.zero_grad()
                #(output of autoencoder, feature vector of input, feature vector of rotated data)
                output, f_data, f_targets = model(data, targets,angles) 

                #Loss
                loss,reconstruction_loss,penalty_loss=penalised_loss(args,output,targets,f_data,f_targets,prop, Lambda)

                # Backprop
                loss.backward()
                optimizer.step()

                #Log progress
                
    
        #Save losses
        sys.stdout.write('Starting evaluation \n')
        sys.stdout.flush()
        mean, std= get_error_per_digit(args,model,args.batch_size_eval,args.step,args.digit,prop)

        average_abs_error[model_name]=mean
        error_std[model_name]=std


    #Change index
    average_abs_error.index=average_abs_error.index* args.step
    error_std.index=error_std.index* args.step

    #Get area under the curve
    AUC=np.zeros(args.samples)
    samples=np.array(samples)
    AUC_df=pd.DataFrame({'prop':samples[:,0], 'Lambda':samples[:,1]})

    for idx, column in  enumerate(average_abs_error.columns):
        y=average_abs_error[column]
        y[0]=0.0
        AUC[idx]=np.trapz(y,average_abs_error.index)

    AUC_df['AUC']=AUC
    average_abs_error.to_csv(os.path.join(path,'average_abs_error.csv'))
    error_std.to_csv(os.path.join(path,'error_std.csv'))
    AUC_df.to_csv(os.path.join(path,'AUC.csv'))



def get_sample(prop_range=(0.01,0.5),Lambda_range=(1,5)):
    """
    Returns a sample from the bivariate uniform distribution
    of prop and lambda

    Args:
        prop_range (tuple) (Lower bound, Upper Bound)
       §Lambda_range (tuple) (Lower bound, Upper Bound)

    Output:
        Tuple
    """
    low=[prop_range[0],Lambda_range[0]]
    high= [prop_range[1],Lambda_range[1]]

    samples=np.random.uniform(low,high)

    return (float(samples[0]),float(samples[1]))


if __name__ == '__main__':
    main()
