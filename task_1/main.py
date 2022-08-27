# +
# # !pip install ipywidgets
# -

import warnings
warnings.filterwarnings("ignore")
import sys
sys.argv = ['']

import argparse
import math
import argparse
import math
import os
import pathlib
import time
import pandas as pd

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy,MetricsCalculator, create_dir,save_history, save_img_samples
from model.wrn import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn.functional as F


def main(args,thrs):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    
    #torch.cuda.empty_cache()
    print('dataset: {}'.format(args.dataset))
    print('samples: {}'.format(args.num_labeled))
    print('model depth-width: {} - {}'.format(args.model_depth,args.model_width))
    print('Threshold: {}'.format(thrs))
    task = "task_1_{}_{}_thres_{}".format(args.dataset,args.num_labeled,thrs)
    saved_weights_dir = os.path.join(args.output_path,task,'weights')
    csv_dir = os.path.join(args.output_path,task,'train_csv')
    img_dir = os.path.join(args.output_path,task,'image')
    
    create_dir(saved_weights_dir)
    create_dir(csv_dir)
    create_dir(img_dir)    
    criterion = torch.nn.CrossEntropyLoss()    

    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
    loss_history = {"epoch": [], "train loss": []}
    acc_history = {"epoch":[], "train acc":[]}    

    for epoch in range(args.epoch):
        
        epoch_start_time = time.time()
        
        epoch_train_loss = MetricsCalculator()
        epoch_acc = MetricsCalculator()

        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            optimizer.zero_grad()
            ##forward pass          
            outputs_predict = model(x_l)   ##predicted on labeled values
            lab_loss = criterion(outputs_predict, y_l.long()) #loss observed on predicted values
            
            ##adding the pseudo labels
            model.eval()
            pseudo_lbl_pred = model(x_ul)
            pseudo_class_prob = torch.softmax(pseudo_lbl_pred, dim=-1)
            max_prob,indices = torch.max(pseudo_class_prob,dim=-1)
            mask = max_prob.ge(thrs).float()
            
            model.train() 
#             indices_add = indices*mask
#             x_11 = torch.flatten(x_ul,start_dim=1)
#             m1 = mask.unsqueeze(dim=1)
#             kt = torch.mul(x_11,m1)
#             ulb_add = torch.reshape(kt,(x_ul.shape[0],x_ul.shape[1],x_ul.shape[2],x_ul.shape[3]))
#             pseudo_set = pseudo_set + tuple((ulb_add, indices_add))
            u_lbl_pred = model(x_ul)
    
            ul_loss = (F.cross_entropy(u_lbl_pred, indices.long(),reduction='none') * mask).mean()
            
            loss = lab_loss + ul_loss
            
            ##backward pass
            loss.backward()
            optimizer.step()
            
            
            acc_score = accuracy(outputs_predict, y_l)
                        
            epoch_train_loss.update(loss.detach().item())
            epoch_acc.update(acc_score[0].detach().item())
        
            # scores summation and writing, update, printing, weights save
        state = {'epoch': (epoch+1),'state_dict': model.state_dict(),
                 'optimizer':  optimizer.state_dict(),'loss': epoch_train_loss.avg}

        if epoch%5==0 :
            torch.save(state, os.path.join(saved_weights_dir, 'model_{}.pt'.format(epoch+1)))
            print("weights saved at epoch {}".format(epoch+1))

        loss_history["epoch"].append(epoch+1)
        loss_history["train loss"].append(epoch_train_loss.avg)

        acc_history["epoch"].append(epoch+1)
        acc_history["train acc"].append(epoch_acc.avg)
        
        epoch_end_time = time.time() - epoch_start_time
        #### Print statement on the screen 
        print("\nEpoch: {}/{}: Time taken: {}".format(epoch+1, args.epoch, epoch_end_time))
        print ('Epoch: {}/{}: Train Loss: {:.4f}'.format(epoch+1, args.epoch, epoch_train_loss.avg))
        print ('Epoch: {}/{}: Train Accuracy: {:.4f}'.format(epoch+1, args.epoch, epoch_acc.avg))
        print("-"*100)

    pd.DataFrame.from_dict(data=acc_history, orient='columns').to_csv(os.path.join(csv_dir,"accuracy.csv"), header=["epoch","train acc"])
    pd.DataFrame.from_dict(data=loss_history, orient='columns').to_csv(os.path.join(csv_dir,"loss.csv"), header=["epoch", "train loss"])
    
    torch.save(model.state_dict(), os.path.join(saved_weights_dir, 'full_model.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=250, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*100, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
#     parser.add_argument('--threshold', type=float, default=0.95,
#                         help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    
    parser.add_argument('--output-path', default= "./output", type=str,
                        help='path to the model weights, csvs, saved imgs')
    
    args = parser.parse_args()
    for thres in [0.95, 0.75, 0.6]:
        main(args,thres)


