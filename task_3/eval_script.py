import sys
sys.path.insert(0, '/netscratch/devidas/nnti/task_3')

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
from utils      import accuracy,MetricsCalculator, create_dir,save_history, save_test_samples, save_img_samples
from model.wrn  import WideResNet
from model.wrn_new  import Proposed
from test       import test_cifar10, test_cifar100


# +
def eval(args):
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
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)

#     summary(model,(3,32,32))
    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################

    torch.cuda.empty_cache()
    total_start_time = time.time()
    
    print('dataset: {}'.format(args.dataset))
    print('samples: {}'.format(args.num_labeled))
    print('vat_eps: {}'.format(args.vat_eps))    
    
    task = "task_3_{}_{}".format(args.dataset,args.num_labeled)
    model_weight = 'model_506.pt'
    
    saved_weights_dir = os.path.join(args.output_path,task,'weights')
    csv_dir = os.path.join(args.output_path,task,'test_csv')
    img_dir = os.path.join(args.output_path,task,'test_samples')
    
#     create_dir(saved_weights_dir)
    create_dir(csv_dir)
    create_dir(img_dir)
    
    test_weight = os.path.join(saved_weights_dir,model_weight)

    #======================================================================================
    # Metrices dictionary
    #======================================================================================
    acc_history = {"Weight":[],"Time":[], "Top 1 acc":[], "Top 5 acc":[]}
    acc_top1 = MetricsCalculator()
    acc_top5 = MetricsCalculator()

    with torch.no_grad():
        for batch_idx, (x_l, y_l) in enumerate(test_loader):
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            
            if args.dataset == "cifar10":
                pred_prob = test_cifar10(x_l, args, filepath = test_weight)
            elif args.dataset == "cifar100":
                pred_prob = test_cifar100(x_l, args, filepath = test_weight)            
           
            acc_score_1, acc_score_5 = accuracy(pred_prob, y_l,topk=(1,5))

            acc_top1.update(acc_score_1[0].detach().item())
            acc_top5.update(acc_score_5[0].detach().item())
            
            if batch_idx == 10:
                save_test_samples(x_l.detach().cpu(), y_l.detach().cpu(), torch.argmax(pred_prob,dim=1).detach().cpu(),img_dir)

        acc_history["Top 1 acc"].append(acc_top1.avg)
        acc_history["Top 5 acc"].append(acc_top5.avg)
        acc_history["Weight"].append(model_weight)
        
        epoch_start_time = time.time()
        epoch_end_time = epoch_start_time - total_start_time
        
        acc_history["Time"].append(epoch_end_time)
        #### Print statement on the screen 

        print("\nTime taken: {:.4f}".format(epoch_end_time))
        print ("Test: Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f}".format(acc_top1.avg,acc_top5.avg))
        print("-"*60)

    pd.DataFrame.from_dict(data=acc_history, orient='columns').to_csv(os.path.join(csv_dir,'test_accuracy.csv'), header=['Weight','Time','Top 1 acc','Top 5 acc'])


# -

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
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
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=10.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=5, type=int, 
                        help="VAT iteration parameter") 
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')                            
    parser.add_argument('--output-path', default= "./output", type=str,
                        help='path to the model weights, csvs, saved imgs')
    
    args = parser.parse_args()

    eval(args)


