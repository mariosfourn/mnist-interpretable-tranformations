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

import matplotlib
from scipy.ndimage.interpolation import rotate
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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
        nrow = int(np.floor(np.sqrt(images.size(0))))

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


def penalised_loss(args,output,targets,f_data,f_targets):
    """
    Define penalised loss
    """

    # Binary cross entropy loss
    loss_fnc = nn.BCELoss(size_average=True)
    loss_reg = Penalty_Loss(size_average=True,proportion=args.prop,type=args.loss)
    #Add 
    reconstruction_loss=loss_fnc(output,targets)
    rotation_loss=loss_reg(f_data,f_targets)
    total_loss= reconstruction_loss+args.Lambda*rotation_loss
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
 


def rotation_test(args, model, device, test_loader):
    """
    Test how well the eoncoder discrimates angles
    return the average error and std in degrees
    """
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            

            target,angles = rotate_tensor(data.numpy())
            angles=angles.reshape(-1,1)
            data=data.to(device)
            target = torch.from_numpy(target).to(device)
            
            #Get Feature vector for original and tranformed image

            x=model.encoder(data) #Feature vector of data
            y=model.encoder(target) #Feature vector of targets

            #Compare Angles            
            x=x.view(x.shape[0],-1) # collapse 3D tensor to 2D tensor 
            y=y.view(y.shape[0],-1) # collapse 3D tensor to 2D tensor
           

            #Number of features
            total_dims=x.shape[1]
            #Batch size
            batch_size=x.shape[0]

            #Number of features penalised
            ndims=round_even(self.proportion*total_dims)  
            #Loop every 2 dimensions
            for i in range(0,ndims-1,2):
                x_i=x[:,i:i+2]      
                y_i=y[:,i:i+2]
                #Get dor product for the batcg
                dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)

                #Get euclidean norm
                x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
                y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

                #Get the cosine of the angel for example
                angles_estimate+=dot_prod/(x_norm*y_norm)

            angles_estimate=torch.acos(angles_estimate/(ndims//2))*180/np.pi # average and in degrees
            angles_estimate=angles_estimate.cpu()
            error=angles_estimate.numpy()-(angles*180/np.pi)
            average_error=error.mean()
            error_std=error.std(ddof=1)

            break
    return average_error,error_std


def read_idx(filename):
    import struct
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def get_metrics(model, data_loader,device, step=5):
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
            ndims=round_even(self.proportion*total_dims)  
            
            sys.stdout.write("\r%d%% complete" % ((batch_idx * 100)/len(data_loader)))
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


def get_error_per_digit(path,batch_size=100, step=5):
    """
    Return plots and csv files with the mean error per MNIST digit in the trainign dataset 
    for a range of rotation 
    Args:
        step (scalar):  rotation step in degrees
    """

    #Load Dataset for each MNIST digit
    data_loaders={digit:DataLoader (MNISTDadataset('./data/',digit), 
        batch_size=args.batch_size, shuffle=False, **kwargs) for digit in range(0,10)}

    #Set up DataFrames
    #mean_error = pd.DataFrame()
    mean_abs_error=pd.DataFrame()
    error_std=pd.DataFrame()

    for digit, data_loader in data_loaders.items():
        sys.stdout.write('Processing digit {} \n'.format(digit))
        sys.stdout.flush()
        results=get_metrics(model,data_loader,device,step)
        #mean_error[digit]=pd.Series(results[0])
        mean_abs_error[digit]= pd.Series(results[1])
        error_std[digit]= pd.Series(results[2])

    #mean_error.index=mean_error.index*step
    mean_abs_error.index=mean_abs_error.index*step
    error_std.index=error_std.index*step

    #mean_error.to_csv(os.path.join(path, 'mean_error_per.csv')
    mean_abs_error.to_csv(os.path.join(path,'mean_abs_error_per_digit.csv'))
    error_std.to_csv(os.path.join(path,'error_std_per_digit.csv'))

        ##Plottin just absolute error
    with plt.style.context('ggplot'):
        mean_abs_error.plot(figsize=(9, 8))
        plt.xlabel('Degrees')
        plt.ylabel('Average error in degrees')
        plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
                   ncol=2, shadow=True, title="Digits", fancybox=True)
        
        plt.tick_params(colors='gray', direction='out')
        plt.savefig(os.path.join(path,'Abs_mean_curves_per_digit.png'))
        plt.close()

    ##Plotting absoltue error and std
    with plt.style.context('ggplot'):
        fig =plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        x=mean_abs_error.index
        for digit in mean_abs_error.columns:
            mean=mean_abs_error[digit]
            std=error_std[digit]
            line,= ax.plot(x,mean)
            ax.fill_between(x,mean-std,mean+std,alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Average error in degrees')
        ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
                   ncol=2, shadow=True, title="Digits", fancybox=True)
        ax.tick_params(colors='gray', direction='out')
        fig.savefig(os.path.join(path,'Abs_mean_&_std_per_digit.png'))
        fig.clf()

def main():

    # Training settings
    list_of_choices=['mse','abs']
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size-recon', type=int, default=10, metavar='N',
                        help='input batch size for reconstruction testing (default: 10)')
    parser.add_argument('--test-batch-size-rot', type=int, default=1000, metavar='N',
                        help='input batch size for rotation disrcimination testing (default: 1,000)')
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
    parser.add_argument('--store-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--Lambda',type=float, default=0.1, metavar='Lambda',
                        help='proportion of penalty loss of the total loss (default=0.1)')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument('--prop',type=float, default=1.0,
                        help='proportion of feature vector with penalty loss')
    parser.add_argument("--loss",dest='loss',default='mse',
    choices=list_of_choices, help='Decide type of penatly loss, mse (defautl) or abs')  
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

    # Init model and optimizer
    model = Net_Reg(device).to(device)
  
    #Initialise weights
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    prediction_avarege_error=[] #Average  rotation prediction error in degrees
    prediction_error_std=[] #Std of error for rotation prediciton
    recon_train_loss=[] # Reconstruction traning loss
    penalty_train_loss=[] # Penalty loss during training

    # Where the magic happens
    for epoch in range(1, args.epochs + 1):
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
            loss,reconstruction_loss,penalty_loss=penalised_loss(args,output,targets,f_data,f_targets)

            # Backprop
            loss.backward()
            optimizer.step()

            #Log progress
            if batch_idx % args.log_interval == 0:
                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                    .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            #Store training and test loss
            if batch_idx % args.store_interval==0:
                #Evaluate loss in 1,0000 sample of the traning set
        
                recon_loss, pen_loss=evaluate_model(args,device,model,train_loader_rotation)
                recon_train_loss.append(recon_loss.item())
                penalty_train_loss.append(pen_loss.item())

                # Average prediction error in degrees
                average, std=rotation_test(args, model, device,train_loader_rotation)
                prediction_error.append(average)
                prediction_error_std.append(std)

        
        if epoch % 5==0:
            #Test reconstruction by printing image
            reconstruction_test(args, model, device,train_loader_recon, epoch)
    #Save model
    save_model(args,model)
    #Save losses
    recon_train_loss=np.array(recon_train_loss)
    penalty_train_loss=np.array(penalty_train_loss)
    prediction_average_error=np.array(prediction_error)
    prediction_error_std=np.array(rediction_error_std)

    learning_curves_DataFrame=pd.DataFrame()

    learning_curves_DataFrame['BCE Training Loss']=recon_train_loss
    learning_curves_DataFrame['Penalty Training Loss']=penalty_train_loss
    learning_curves_DataFrame['Average Abs error']=prediction_average_error
    learning_curves_DataFrame['Error STD']=prediction_average_error

    learning_curves_DataFrame.index=learning_curves_DataFrame.index*args.store_interval*args.batch_size

    learning_curves.to_csv(os.path.join(path,'learning_curves.csv'))

    plot_learning_curve(args,recon_train_loss,penalty_train_loss,prediction_average_error, prediction_error_std,path)

    get_error_per_digit(path,batch_size=100, step=5)



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
        ax2.fill_between(x_ticks,average_error+error_std,average_error+error_std,
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
