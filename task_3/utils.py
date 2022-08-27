import torch
import pathlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#======================================================================================
# For avging metrics values
#======================================================================================
class MetricsCalculator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


#======================================================================================
# Copy all output data to CSV files. Can be used for plotting later on..
#======================================================================================
def save_history(loss_data, acc_data,path_dir): 
#     create_dir(path_dir)
    pd.DataFrame.from_dict(data=loss_data, orient='columns').to_csv(os.path.join(path_dir,'loss.csv'), header=['epoch','train loss','vat loss','cl loss'])
    pd.DataFrame.from_dict(data=acc_data, orient='columns').to_csv(os.path.join(path_dir,'accuracy.csv'), header=['epoch','train acc'])


# +
def save_test_samples(inp, out, pred, save_dir):
    
    imgs_per = inp.permute(0,2,3,1).numpy()
    out = out.numpy()
    pred = pred.numpy()

#     # take the images data from batch data
#     images = data_batch_1['data']
#     # reshape and transpose the images
#     images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
#     # take labels of the images 
#     labels = data_batch_1['labels']
#     # label names of the images
#     label_names = meta_data['label_names']


    # dispaly random images
    # define row and column of figure
    rows, columns = 5, 5
    ## take random image idex id
#     imageId = np.random.randint(0, len(images), rows * columns)
#     # take images for above random image ids
#     images = images[imageId]
#     # take labels for these images only
#     labels = [labels[i] for i in imageId]

    # define figure
    fig=plt.figure(figsize=(10, 10))
    # visualize these random images
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs_per[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title("True: {} | Pred: {}".format(out[i-1],pred[i-1]))
    fig.tight_layout()
#     plt.show()
    plt.savefig(os.path.join(save_dir, 'sample_{}.png'.format(10)), dpi=500)
    
def save_img_samples(perturb_img,unperurb_img,save_dir,batch_size):

    imgs_per = perturb_img.permute(0,2,3,1).numpy()
    imgs_unper = unperurb_img.permute(0,2,3,1).numpy()


    f, ax1 = plt.subplots(5,2,figsize=(15,15))
    ax1[0][0].set_title('Unperturbed Image')
    ax1[0][1].set_title('Perturbed Image')
  
    for j in range(5):

        ax1[j][0].imshow(imgs_unper[j])
        ax1[j][0].set_xticklabels([])
        ax1[j][0].set_yticklabels([])
        
        ax1[j][1].imshow(imgs_per[j])
        ax1[j][1].set_xticklabels([])
        ax1[j][1].set_yticklabels([])

    plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    f.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_{}.png'.format(10)), dpi=500)

