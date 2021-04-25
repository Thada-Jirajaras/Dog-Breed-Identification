!!!!This report is just the template and still in development process. To see some codes and description please look at **"dog_app.ipynb"**

## Using Convolutional Neural Networks to Identify Dog Breeds
Thada Jirajaras  
May 31st, 2021

## I. Definition
### Project Overview
Nowaday, AI and computer vision play an important role in detecting objects [1]. There are many models pretrained with a big dataset for this task. For example, there are models that can identify objects in the ImageNet dataset which have 1000 classes [2].

Howevers, there are limitations that these pretrained models can identify only the classes that appear in the trained dataset. For example, in order to specifically identify unseen classes like some dog breeds, the models need to be newly created. Also, there may be situations where there are only a few dog images that have the breed labels because it requires a lot of effort to label images [3]. Thus, the model must use only a few images to train.

### Problem Statement
With only a few dog images with breed labels, it is challenging to create a model that does not overfit the training set and give high accuracy on the test set.

### Metrics
Accuracy is the acceptable and selected metric because classes are not extremely unbalanced.


## II. Analysis
### Data Exploration
Dog dataset: (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) This is the main dataset for developing a CNN model to identify dog breeds. The dataset contains dog images with 133 breed labels. The dataset consist of training (6680 images), validation (835 images), and test (836 images) sets
Random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.

![train_distribution](./resultimages/train_distribution.jpg)

![valid_distribution](./resultimages/valid_distribution.jpg)

![test_distribution](./resultimages/test_distribution.jpg)

### Exploratory Visualization

Assigning breed to dogs from images is considered exceptionally challenging. To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.

|                  **Brittany**                  |                  **Welsh Springer Spaniel**                  |
| :--------------------------------------------: | :----------------------------------------------------------: |
| ![Brittany_02625](./images/Brittany_02625.jpg) | ![Welsh_springer_spaniel_08203](./images/Welsh_springer_spaniel_08203.jpg) |



It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels). 

|                  **Curly-Coated Retriever**                  |                  **American Water Spaniel**                  |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![Curly-coated_retriever_03896](./images/Curly-coated_retriever_03896.jpg) | ![American_water_spaniel_00648](./images/American_water_spaniel_00648.jpg) |

Likewise, recall that labradors come in yellow, chocolate, and black. Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.

|                       Yellow Labrador                        |                      Chocolate Labrador                      |                        Black Labrador                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![Labrador_retriever_06457](./images/Labrador_retriever_06457.jpg) | ![Labrador_retriever_06455](./images/Labrador_retriever_06455.jpg) | ![Labrador_retriever_06449](./images/Labrador_retriever_06449.jpg) |



### Algorithms and Techniques

In order to achieve high accuracy from fitting the model with only a few images, transfer learning must be used. Also, the chosen base models should be trained on the dataset that contain some different dog breeds. Fortunately, all base models from torchvision are trained on the ImageNet dataset which contains some image categories corresponding to dogs [2]. Thus, one of these base models will be chosen for transfer learning in this project.

### Benchmark
There are some works that use the same datasets to create and test the models [4, 5] and give the similar benchmark results. For example, Maanav Shah created 4 models including VGG-49, ResNet-50, Inception V3, and Xception [4]. The model performance comparisons of his work are as follows.

| Model        | Acc  | Training Time in seconds <br />(20 epochs, batch size = 32) |
| ------------ | ---- | ----------------------------------------------------------- |
| VGG-19       | 46%  | 24.16                                                       |
| ResNet-50    | 82%  | 18.41                                                       |
| Inception v3 | 81%  | 31.83                                                       |
| Xception     | 85%  | 49.07                                                       |




## III. Methodology
### Data Preprocessing
#### Image transformations for training set

1. transforms.Resize((244, 244))
2. transforms.RandomHorizontalFlip()
3. transforms.RandomRotation(10),
4. transforms.ToTensor()
5. transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

#### Image transformations for evaluation and test sets

1. transforms.Resize((244, 244))
2. transforms.ToTensor()
3. transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Implementation

ImageNet50 are chosen as a pretrained model. The way to use transferred model with the new dataset is to freeze all feature layers and only allow classification layer to be trained. The steps are as follows.

1. Load the pretrained model "restnet50" for the ImageNet dataset from torchvision
2. Freeze all features layers not to be trained
3. Create new classification layer to have the number of output nodes equal to the number of dog breeds which is 133
4. Train the parameters of the classification layer

After 19 epochs of refinement training, the best model with lowest loss in validation set provides 82% of accuracy  in the test set.  Later iterations give insignificant improvement in the validation loss. Thus, some hyperparameter may be changed before continue training the model.  One possible parameter that can affect the training performance after training for some epochs and need to be adapted is the learning rate.

### Refinement

Model refinement is made by reducing the learning rate from 0.001 to 0.0001. The steps are as follows. 

1. Load the best trained model 
2. Continue training with lower learning rate (lr = 0.0001)

After 25 epochs of refinement training, the best model with lowest loss in validation set provides 86% of accuracy  in the test set. The accuracy increases 4% from the model without refinement


## IV. Results
### Model Evaluation and Validation
The best model provide the 86% of the accuracy in test set.Justification

The ResNet-50 model after refinement give higher accuracy (86%) than the accuracy  of ResNet-50 (82%) and the accuracy of Xception (85%) provided in the benchmark section. However, if more advance learning rate policy is applied, the model may provide even higher accuracy [4].


## V. Conclusion
Transfer learning can help identification problems that have only a few image samples to train the new model. In this project, the transferred model provides high accuracy (86%) in the test set.   

Dog-breed identification can be a part of some application. For example, the application can accepts a file path to an image and first determines whether the image contains a human, dog, or neither based on  Haar feature-based cascade classifiers to detect human faces and pretrain VGG16 to detect dogs. Then, if a dog is detected in the image, dog-breed identification can be used to return the predicted breed. if a human is detected in the image, return the resembling dog breed. if neither is detected in the image, provide output that indicates an error. Some results of this application are shown as follows.

| ![dog1](./resultimages/dog1.jpg) | ![human1](./resultimages/human1.jpg) |
| :------------------------------: | :----------------------------------: |
| ![dog2](./resultimages/dog2.jpg) | ![human2](./resultimages/human2.jpg) |
| ![dog3](./resultimages/dog3.jpg) | ![human3](./resultimages/human3.jpg) |

### 

### Reflection
- Setting "Shuffle = True" in the data loader for the train set is very important.  If we set Shuffle = False, it is possible that the samples in the same batch contains only one class or only a few classes and cause model to not be able to learn much from that batch.
- Even though, some acceptable results are met, we may be able to improve the performance of the result further by adjusting only one parameter or only a few parameters. For example, after the model can achieve 82% of the accuracy it can be improved more by only reducing the learning rate parameter and continuing training.

### Improvement
To improve the result, these factors may need to explored further

- Learning rate policies
- Pretrained model architectures  
- Batch size

## Reference

1. Medium. 2021. Everything You Ever Wanted To Know About Computer Vision. Here’s A Look Why It’s So Awesome.. [online] Available at:
<https://towardsdatascience.com/everything-you-ever-wanted-to-know-about-computer-vision-heres-a-look-why-it-s-so-awesome-e8a58dfb641e> [Accessed 20 April 2021].
2. Pytorch.org. 2021. torchvision.models — Torchvision master documentation. [online]
Available at: <https://pytorch.org/vision/stable/models.html> [Accessed 20 April 2021].
3. Anolytics. 2021. Top Data Labeling Challenges Faced by the Data Annotation Companies. [online] Available at:
<https://www.anolytics.ai/blog/top-data-labeling-challenges-faced-by-annotation-companies/>
[Accessed 20 April 2021].
4. Medium. 2021. Dog Breed Classifier using CNN. [online] Available at:
    <https://medium.com/@maanavshah/dog-breed-classifier-using-cnn-f480612ac27a>
    [Accessed 20 April 2021].
5. Medium. 2021. Deep Learning: Build a dog detector and breed classifier using CNN. [online] Available at:
    <https://towardsdatascience.com/deep-learning-build-a-dog-detector-and-breed-classifier-usi
    ng-cnn-f6ea2e5d954a> [Accessed 20 April 2021].
6. Ayanzadeh, Aydin & Vahidnia, Sahand. (2018). Modified Deep Neural Networks for Dog Breeds Identification.