import argparse
import math
import os
import pathlib
import time

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
# from torchsummary import summary

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy,MetricsCalculator, create_dir,save_history, save_img_samples
from model.wrn  import WideResNet


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
    task = "task_2_{}_{}_eps_{}".format(args.dataset,args.num_labeled,int(args.vat_eps))
    
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
    loss_history = {"epoch": [], "train loss": [], "vat loss":[], "cl loss":[]}
    acc_history = {"epoch":[], "train acc":[]}

    for epoch in range(args.epoch):
        
        epoch_start_time = time.time()
        
        epoch_vat_loss = MetricsCalculator()
        epoch_cl_loss = MetricsCalculator()
        epoch_train_loss = MetricsCalculator()
        epoch_acc = MetricsCalculator()

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
            # TODO: SUPPLY you code
            ####################################################################

            optimizer.zero_grad()
            VT_LOSS = VATLoss(args)
            vt_loss, perturb_inp = VT_LOSS(model,x_ul)
            predictions = model(x_l)
            cl_loss  = criterion(predictions,y_l.long())
            loss = cl_loss + args.alpha * vt_loss

            loss.backward()
            optimizer.step()
            acc_score = accuracy(predictions, y_l)
            
            if epoch%32==0 and i==0:
                save_img_samples(perturb_inp.detach().cpu(),x_ul.detach().cpu(),img_dir,args.train_batch)
            
            epoch_train_loss.update(loss.detach().item())
            epoch_vat_loss.update(vt_loss.detach().item())
            epoch_cl_loss.update(cl_loss.detach().item())

            epoch_acc.update(acc_score[0].detach().item())
            
            # scores summation and writing, update, printing, weights save
        state = {'epoch': (epoch+1),'state_dict': model.state_dict(),
                 'optimizer':  optimizer.state_dict(),'loss': epoch_train_loss.avg}

        if epoch%5==0 :
            torch.save(state, os.path.join(saved_weights_dir, 'model_{}.pt'.format(epoch+1)))
            print("weights saved at epoch {}".format(epoch+1))

        loss_history["epoch"].append(epoch+1)
        loss_history["vat loss"].append(epoch_vat_loss.avg)
        loss_history["cl loss"].append(epoch_cl_loss.avg)
        loss_history["train loss"].append(epoch_train_loss.avg)

        acc_history["epoch"].append(epoch+1)
        acc_history["train acc"].append(epoch_acc.avg)
        
        epoch_end_time = time.time() - epoch_start_time
        #### Print statement on the screen 
        print("\nEpoch: {}/{}: Time taken: {}".format(epoch+1, args.epoch, epoch_end_time))
        print ('Epoch: {}/{}: Vat Loss: {:.4f} Cl Loss: {:.4f} Train Loss: {:.4f}'.format(epoch+1, args.epoch, epoch_vat_loss.avg, epoch_cl_loss.avg, epoch_train_loss.avg))
        print ('Epoch: {}/{}: Train Accuracy: {:.4f}'.format(epoch+1, args.epoch, epoch_acc.avg))
        print("-"*100)

    save_history(loss_history, acc_history, csv_dir)
    torch.save(model.state_dict(), os.path.join(saved_weights_dir, 'full_model.pt'))

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
    parser.add_argument('--output-path', default= "./output", type=str,
                        help='path to the model weights, csvs, saved imgs')
    
    args = parser.parse_args()

    main(args)


