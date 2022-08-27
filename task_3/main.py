import argparse
import math
import os
import pathlib
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import sys
sys.argv = ['']

# +
# # !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# # !pip install ipywidgets
# # !pip install torchsummary
# torch.__version__
# -

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn.functional as F

# from torchsummary import summary

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy,MetricsCalculator, create_dir,save_history, save_img_samples
from model.wrn  import WideResNet
from model.wrn_new  import Proposed
# from test       import test_cifar10, test_cifar100

def main(args):
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
    if args.model_name=="Proposed":
        model       = Proposed(args.model_depth, 
                                    args.num_classes, widen_factor=args.model_width)
    elif args.model_name=="FixMatch":
        model       = WideResNet(args.model_depth, 
                                    args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)
#     summary(model,(3,32,32))
    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    torch.cuda.empty_cache()
    total_start_time = time.time()
    
    print('dataset: {}'.format(args.dataset))
    print('samples: {}'.format(args.num_labeled))
    print('vat_eps: {}'.format(args.vat_eps))
    print('model depth-width: {} - {}'.format(args.model_depth,args.model_width))
#     print('old')
    task = "task_3_{}_{}".format(args.dataset,args.num_labeled)
    
    saved_weights_dir = os.path.join(args.output_path,task,'weights')
    csv_dir = os.path.join(args.output_path,task,'train_csv')
    img_dir = os.path.join(args.output_path,task,'image')
    
    create_dir(saved_weights_dir)
    create_dir(csv_dir)
    create_dir(img_dir)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #======================================================================================
    # Metrices dictionary
    #======================================================================================
    loss_history = {"epoch": [], "train loss": [], "labeled loss":[], "unlabeled loss":[]}
    acc_history = {"epoch":[], "train acc":[]}
    mask_probabilities = {"epoch":[], "Mask Probs":[]}
    for epoch in range(args.epoch):
        
        epoch_start_time = time.time()
        
        epoch_u_l_loss = MetricsCalculator()
        epoch_x_l_loss = MetricsCalculator()
        epoch_train_loss = MetricsCalculator()
        epoch_acc = MetricsCalculator()
        epoch_mask_probs =  MetricsCalculator()

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
                x_ul_weak,x_ul_strong, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul_weak,x_ul_strong, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul_weak, x_ul_strong        = x_ul_weak.to(device), x_ul_strong.to(device)
            ####################################################################
            # TODO: SUPPLY you code
            ####################################################################

            optimizer.zero_grad()

            pred_weak = model(x_ul_weak)
            pred_strong = model(x_ul_strong)
            pred_x_l = model(x_l)

            cl_loss_x_l  = criterion(pred_x_l,y_l.long())

            pseudo_lbl = torch.softmax(pred_weak.detach()/args.T, dim=-1)

            max_probs, targets_u_l = torch.max(pseudo_lbl, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            cl_loss_u_l = (F.cross_entropy(pred_strong, targets_u_l.long(),
                                  reduction='none') * mask).mean()

            loss_cosine_sim = 1.0 - F.cosine_similarity(torch.softmax(pred_strong.detach()/args.T, dim=-1), pseudo_lbl).mean()
            
            oss = cl_loss_x_l + args.lambda_u * cl_loss_u_l + args.rho_cs * loss_cosine_sim

            loss.backward()
            optimizer.step()

            
            acc_score = accuracy(pred_x_l, y_l)
                        
            epoch_train_loss.update(loss.detach().item())
            epoch_x_l_loss.update(cl_loss_x_l.detach().item())
            epoch_u_l_loss.update(cl_loss_u_l.detach().item())

            epoch_acc.update(acc_score[0].detach().item())

            epoch_mask_probs.update(mask.mean().item())
            
            # scores summation and writing, update, printing, weights save
        state = {'epoch': (epoch+1),'state_dict': model.state_dict(),
                 'optimizer':  optimizer.state_dict(),'loss': epoch_train_loss.avg}

        if epoch%5==0 :
            torch.save(state, os.path.join(saved_weights_dir, 'model_{}.pt'.format(epoch+1)))
            print("weights saved at epoch {}".format(epoch+1))

        loss_history["epoch"].append(epoch+1)
        loss_history["labeled loss"].append(epoch_x_l_loss.avg)
        loss_history["unlabeled loss"].append(epoch_u_l_loss.avg)
        loss_history["train loss"].append(epoch_train_loss.avg)

        acc_history["epoch"].append(epoch+1)
        acc_history["train acc"].append(epoch_acc.avg)
        
        mask_probabilities["epoch"].append(epoch+1)
        mask_probabilities["Mask Probs"].append(epoch_mask_probs.avg)

        epoch_end_time = time.time() - epoch_start_time
        #### Print statement on the screen 
        print("\nEpoch: {}/{}: Time taken: {}".format(epoch+1, args.epoch, epoch_end_time))
        print ('Epoch: {}/{}: Labeled Loss: {:.4f} Unlabeled Loss: {:.4f} Train Loss: {:.4f}'.format(epoch+1, args.epoch, epoch_x_l_loss.avg, epoch_u_l_loss.avg, epoch_train_loss.avg))
        print ('Epoch: {}/{}: Train Accuracy: {:.4f}'.format(epoch+1, args.epoch, epoch_acc.avg))
        print("-"*100)

    pd.DataFrame.from_dict(data=acc_history, orient='columns').to_csv(os.path.join(csv_dir,"accuracy.csv"), header=["epoch","train acc"])
    pd.DataFrame.from_dict(data=loss_history, orient='columns').to_csv(os.path.join(csv_dir,"loss.csv"), header=["epoch","labeled loss","unlabeled loss","train loss"])
    pd.DataFrame.from_dict(data=mask_probabilities, orient='columns').to_csv(os.path.join(csv_dir,"mask_probs.csv"), header=["epoch","Mask Probs"])
    
    torch.save(model.state_dict(), os.path.join(saved_weights_dir, 'full_model.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
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
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
#     parser.add_argument("--vat-xi", default=10.0, type=float, 
#                         help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=10.0, type=float, 
                        help="VAT epsilon parameter") 
#     parser.add_argument("--vat-iter", default=5, type=int, 
#                         help="VAT iteration parameter") 
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')                            
    parser.add_argument('--rho-cs', default=0.5, type=float,
                        help='coefficient of unlabeled loss') 
    parser.add_argument('--output-path', default= "./output", type=str,
                        help='path to the model weights, csvs, saved imgs')
    parser.add_argument("--model-name", default="Proposed", 
                        type=str, choices=["FixMatch", "Proposed"])  
    args = parser.parse_args()

    main(args)


