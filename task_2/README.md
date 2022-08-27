**For Train:**
Run ```main.py``` for starting the training. But before we run, we need to read the below conditions and perform changes in the script as per your requirements:
1. While training you need to change ```--dataset= 'cifar10' or 'cifar100'``` depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively.
2. To change other hyper parameters such as learning rate, batch size, model depth and weight, labelled samples, dataset etc., we need to change in the argparser arguments. Please refer to the table provided in the report for what parameters were used during training.
3. VAT loss parameters such as ```--vat-xi```, ```--vat-eps``` and ```--vat-iter``` can be changed in the argparser arguments. All these parameters have their default values which we used.
4. We have used ```model depth = 28``` and  ```model width = 2``` for Cifar-10 and Cifar100 dataset. But can be changed in the argparser arguments.

**For Inference:**
We run ```eval_script.py``` for starting the inference. But before running we need to read the below conditions:
1. In the ```eval_script.py``` you need to change ```--dataset=``` to 'cifar10' or 'cifar100' depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively in the argument parser.
2. Remaining all parameters can be changed in the argparser, depending on your choice. But all have their default values and can be used as it is.
