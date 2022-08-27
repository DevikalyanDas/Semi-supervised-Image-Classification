**For Train:**
Run ```main.py``` for starting the training. But before we run, we need to read the below conditions and perform changes in the script as per your requirements:
1. While training you need to change ```--dataset= 'cifar10' or 'cifar100'``` depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively.
2. To change other hyper parameters such as learning rate, batch size, model depth and weight, labelled samples, dataset etc., we need to change in the argparser arguments. Please refer to the table provided in the report for what parameters were used during training.
3. There is no need to provide confidence thresholds as these are passed in the form of a list to the main function and can compute the scores for different thresholds in a loop . If you want to chnage them then you need to edit it at the line (217) where we call the main function.
4. We have used ```model depth = 28``` and  ```model width = 2``` for Cifar-10 and Cifar100 dataset. But can be changed in the argparser arguments.

**For Inference:**
We run ```eval_script.py``` for starting the inference. But before running we need to read the below conditions:
1. In the ```eval_script.py``` you need to change ```--dataset=``` to 'cifar10' or 'cifar100' depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively in the argument parser.
2. Remaining all parameters can be changed in the argparser, depending on your choice. But all have their default values and can be used as it is.
3. There is no need to provide confidence thresholds as these are passed in the form of a list to the eval function and can compute the scores for different thresholds in a loop. If you want to chnage them, then you need to edit it at the line (180) where we call the eval function.
