Team Members:
```Devikalyan Das, 7007352```
```Rushan Mukherjee, 7015520```

Downloadable Link code with weights:
[Google Drive Link](https://drive.google.com/file/d/1p9LrR63veg5FHNMHdnFyruXtASJ3PR8d/view?usp=sharing)

md5sum:
```f5fc4f610060481c5fa982255caa94ce```

This project was carried out for the course Neural Networks:Theory and Implementation 2021/22. <br/>
The goal of this project is to explore different methods of Semi-Supervised Learning (SSL) in order to improve model performance by leveraging information not only from labelled data but also from unlabelled data. This work has been divided into three tasks:<br/>
1. **Pseudo Labelling** : We train our network using our labeled data and then using our trained network, we predict the class  probabilities of our unlabeled data. If the probability of a class is greater than a user defined threshold value for a sample, we assign the sample that particular class and use it for training.
2. **Virtual Adversarial Training** : Here, we train our network with sparsely labeled data (e.g. SSL). In VAT we try to find a perturbation (r) that maximizes the KL divergence between the original image and the adversarial image (by adding noise r to the image). Then, we train the network to predict the same labels for original image and perturbed image.
3. **Challenge Task** : We have to improve the performance of Fix Match. Fix Match is a SSL technique, that uses weak augmentation and strong augmentation applied to the unlabelled images. We train a supervised model on our labeled images with cross-entropy loss. Then we pass the weakly augmented image through the model and obtain the prediction interms of probability and converted to one hot label called pseudo-label. Similarly, the model prediction for strongly augmented images are computed and matched the pseudo label from the weakly augmented images via cross-entropy loss.
