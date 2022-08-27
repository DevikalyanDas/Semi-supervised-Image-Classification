import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn.functional as F


# +
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils      import accuracy,MetricsCalculator, create_dir,save_history, save_test_samples, save_img_samples
from model.wrn  import WideResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -

def test_cifar10(testdataset, args, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # TODO: SUPPLY the code for this function
   
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)
   
    model.eval()
    saved_weights = torch.load(filepath, map_location=torch.device('cpu'))
    
    model.load_state_dict(saved_weights["state_dict"])

    pred_x_l = model(testdataset)

    pred_prob = torch.softmax(pred_x_l.detach()/args.T, dim=-1)    

    return pred_prob

def test_cifar100(testdataset,args, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # TODO: SUPPLY the code for this function
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)
   
    model.eval()
    saved_weights = torch.load(filepath, map_location=torch.device('cpu'))
    
    model.load_state_dict(saved_weights["state_dict"])

    pred_x_l = model(testdataset)

    pred_prob = torch.softmax(pred_x_l.detach()/args.T, dim=-1)    

    return pred_prob
