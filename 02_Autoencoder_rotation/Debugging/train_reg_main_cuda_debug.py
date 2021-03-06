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

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms

from model import Net_Reg

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)


def rotate_tensor(input,rot_range=2*np.pi,plot=False):
    """Nasty hack to rotate images in a minibatch, this should be parallelized
    and set in PyTorch

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


def save_model(args,model,epoch):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    epoch:  trainign epoch
    """

    path='./model_lambda_{}'.format(args.regulariser)
    import os
    if not os.path.exists(path):
      os.mkdir(path)
    if (epoch % 1==0):
        torch.save(model.state_dict(), path+'/checkpoint.pt')


def reconstruction_test(args, model, device, test_loader, epoch):

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            data = data.view(data.size(0), -1)
            data = data.repeat(args.test_batch_size,1)
            data = data.view(args.test_batch_size**2,1, 28,28)
            target = torch.zeros_like(data)

            angles = torch.linspace(0, 2*np.pi, steps=args.test_batch_size)
            angles = angles.view(args.test_batch_size, 1)
            angles = angles.repeat(1, args.test_batch_size)
            angles = angles.view(args.test_batch_size**2, 1)


            # Forward pass
            data = data.to(device)
            target=target.to(device)
            output,_,_ = model(data, target, angles)
            break
        output = output.cpu()
        save_images(args,output, epoch)

def rotation_test(args, model, device, test_loader):
    """
    Test how well the eoncoder discrimates angles
    return the average error in degrees
    """
    model.eval()
    average_error=0.0 #in degrees
    with torch.no_grad():
        for data, target in test_loader:
            #Get rotated vector
            #angles = np.linspace(0,np.pi,test_loader.batch_size)
            target,angles = rotate_tensor(data.numpy(),np.pi,plot=False)
            angles=angles.reshape(-1,1)
            data=data.to(device)
            target = torch.from_numpy(target).to(device)
            
            #Get Feature vector for original and tranformed image

            x=model.encoder(data) #Feature vector of data
            y=model.encoder(target) #Feature vector of targets

            #Compare Angles            
            x=x.view(x.shape[0],-1) # collapse 3D tensor to 2D tensor 
            y=y.view(y.shape[0],-1) # collapse 3D tensor to 2D tensor
            ndims=x.shape[1]        # get dimensionality of feature space
            batch_size=x.shape[0]   # get batch_size
            angles_estimate=torch.zeros(batch_size,1).to(device)   
            for i in range(0,ndims-1,2):
                x_i=x[:,i:i+2]      
                y_i=y[:,i:i+2]
                dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
                x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
                y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)
                angles_estimate+=dot_prod/(x_norm*y_norm)

            angles_estimate=torch.acos(angles_estimate/(ndims//2))*180/np.pi # average and in degrees
            angles_estimate=angles_estimate.cpu()
            average_error+=np.sum((angles*180/np.pi)-angles_estimate.numpy())/len(test_loader.dataset)
    return average_error


def plot_learning_curve(args,recon_train_loss,regulariser_train_loss,rotation_loss):
    """z=
    Plots learning curves at the end of traning
    """
    total_loss=recon_train_loss+args.regulariser*regulariser_train_loss
    x_ticks1=np.arange(len(recon_train_loss))*args.store_interval*arg.batch_size

    fig, ax1=plt.subplots()
    color='tab:red'
    lns1=ax1.plot(x_ticks1,recon_train_loss,label='Training Reconstrunction Loss (BCE)')
    lns2=ax1.plot(x_ticks1,regulariser_train_loss,label='Training Rotation disrcimination loss')
    lns3=ax1.plot(x_ticks1,total_loss,'-.',color=color,label='Total Training Loss')
    ax1.set_xlabel('Training Examples')
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params(axis='y', colors= color)

    color = 'tab:green'
    ax2 = ax1.twinx()
    lns4=ax2.plot(x_ticks1,rotation_loss,color=color,label='Average test rotation loss')
    ax2.set_ylabel('Degrees',color=color)
    ax2.tick_params(axis='y', colors= color)
    ax2

    #Legend
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid(True)
    #ax2.set_ylim([-360,360])
    plt.title(r'Learning Curves with $\lambda$={}'.format(args.regulariser))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    path = "./output_lambda_{}".format(args.regulariser)
    fig.savefig(path+'/learning_curves')
    fig.clf()


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
    path = "./output_lambda_{}".format(args.regulariser)
    plt.savefig(path+"/epoch{:04d}".format(epoch))
    plt.close()

# def copy_weights(model1,model2):
#     """
#     Copies weigths from one model to the other(from model1 to model2)
#     model2 must be contained wihitn model1
#     """
#     model1_dict=model1.state_dict()
#     model2_dict = model2.state_dict()
#     # 1. filter out unnecessary keys
#     model1_dict = {k: v for k, v in model1_dict.items() if k in model2_dict}
#     # 2. overwrite entries in the existing state dict
#     model2_dict.update(model1_dict) 
#     # 3. load the new state dict
#     model2.load_state_dict(model1_dict)

#     return model2



class Reg_Loss(nn.Module):
    
    def __init__(self,size_average=False):
        super(Reg_Loss,self).__init__()
        self.size_average=size_average
        
    def forward(self,x,y):
        """
        regulariser for loss

        Args:
            x: [batch,1,ndims]
            y: [batch,1,ndims]
        """
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        ndims=x.shape[1]
        batch_size=x.shape[0]
        reg_loss=0.0
        for i in range(0,ndims-1,2):
            x_i=x[:,i:i+2]
            y_i=y[:,i:i+2]
            dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
            x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
            y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)
            reg_loss+= (torch.sum(dot_prod/(x_norm*y_norm))-1)**2
        if self.size_average:
            reg_loss=reg_loss/x.shape[0]/(ndims//2)
        return reg_loss


def reg_loss(args,output,targets,f_data,f_targets):
    """
    Define regularised loss
    """

    # Binary cross entropy loss
    loss_fnc = nn.BCELoss(size_average=True)
    loss_reg =Reg_Loss(size_average=True)
    #Add 
    reconstruction_loss=loss_fnc(output,targets)
    rotation_loss=loss_reg(f_data,f_targets)
    total_loss= reconstruction_loss+args.regulariser*rotation_loss
    return total_loss,reconstruction_loss,rotation_loss



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for reconstruction testing (default: 10)')
    parser.add_argument('--test-batch-size-disc', type=int, default=500, metavar='N',
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
    parser.add_argument('--regulariser',type=float, default=0.1, metavar='Lambda',
                        help='regularising constant (default=0.1)')
  
    args = parser.parse_args()

    # Create save path
    path = "./output_lambda_{}".format(args.regulariser)
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

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader_disc = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size_disc, shuffle=False, **{})

    # Init model and optimizer
    model = Net_Reg(device).to(device)
    #Initialise weights and train
    path = "./output_lambda_{}".format(args.regulariser)
  
    #Initialise weights
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Get rotation loss in t
    # rotation_test_loss=rotation_test(args, model_encoder, 'cpu', test_loader_disc)
    rotation_test_loss=[]
    recon_train_loss=[]
    regulariser_train_loss=[]
    loss_log=[]

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
            output, f_data, f_targets = model(data, targets,angles) #for feature vector

            #Gry loss
            #loss,reconstruction_loss,regulariser_loss=reg_loss(args,output,targets,f_data,f_targets)
            loss_fnc = nn.BCELoss(size_average=False)
            loss=loss_fnc(output, targets)

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
                #Train Lossq
                #recon_train_loss.append(reconstruction_loss.item())
                #regulariser_train_loss.append(regulariser_loss.item())
                loss_log.append(loss.item())


                #Test Loss
                #rotation_test_loss.append(rotation_test(args, model, device, test_loader_disc))

        
        if epoch % 5==0:
            #Test reconstruction by printing image
            reconstruction_test(args, model, device, test_loader, epoch)
    #Save model
    save_model(args,model,epoch)
    #Save losses
    # np.save(path+'/recon_train_loss',np.array(recon_train_loss))
    # np.save(path+'/regulariser_train_loss',np.array(regulariser_train_loss))
    # np.save(path+'/rotation_test_loss',np.array(rotation_test_loss))
    # recon_train_loss=np.array(recon_train_loss)
    # regulariser_train_loss=np.array(regulariser_train_loss)
    # rotation_test_loss=np.array(rotation_test_loss)
    # plot_learning_curve(args,recon_train_loss,regulariser_train_loss,rotation_test_loss)

    np.save(path+'loss_log',np.array(loss_log))
    plt.plot(loss_log)
    plt.show()
 
if __name__ == '__main__':
    main()
