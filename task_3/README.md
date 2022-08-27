**For Train:**
Run ```main.py``` for starting the training. But before we run, we need to read the below conditions and perform changes in the script as per your requirements:
1. The same script is used for Fixmatch as well as for Proposed model. If you want to train Fixmatch just need to use ```--model-name = 'FixMatch'``` in the argparser arguments. If you want to train the proposed model, then use ```--model-name = 'Proposed'```.
2. While training you need to change ```--dataset= 'cifar10' or 'cifar100'``` depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively.
3. To change other hyper parameters such as learning rate, batch size, model depth and weight, labelled samples, dataset etc., we need to change in the argparser arguments. Please refer to the tables of parameters provided in the report for what values were used during training.  
4. We have used ```model depth = 28``` and  ```model width = 2``` for Cifar-10 dataset with labelled samples 250 and 4000. You can change them in the argparser arguments.
5. We have used ```model depth = 28``` and  ```model width = 8``` for Cifar-100 dataset with labelled samples 2500 and 10000. You can change them in the argparser arguments.
6. Also, use ```rho-cs = 0```, when using FixMatch and change it to some values between 0 and 1, when using Proposed model. This factor controls the cosine similarity loss function.
7. Also, use ```--output-path=./output``` in case of training fixmatch. While traing proposed model, use ```--output-path=./new_output```.

**For Inference:**
We run ```eval_script.py``` for starting the inference. But before running we need to read the below conditions
1. This script calls the ```test.py``` The same script is used for Fixmatch as well as for Proposed model. If you want to test Fixmatch, then in ```test.py``` just use ```model = WideResNet(args.model_depth,args.num_classes,widen_factor=args.model_width)``` in ```test_cifar10``` or ```test_cifar100``` functions depending on your requirements. 
If you want to test the proposed model, then in ```test.py``` just use ```    model = Proposed(args.model_depth,args.num_classes,widen_factor=args.model_width)``` in ```test_cifar10``` or ```test_cifar100``` functions depending on your requirements.
2. In the ```eval_script.py``` you need to change ```--dataset=``` to 'cifar10' or 'cifar100' depending on your requirement. Then you need to change ```---num-labeled=``` to 4000 or 250 if using 'cifar10' and to 2500 or 10000 if using 'cifar100' above respectively.
3. Also in ```eval_script.py```, use ```--output-path=./output``` in case of testing fixmatch. While inferencing proposed model, use ```--output-path=./new_output```.
