!!!!This report is just the template and still in development process. To see some codes and description please look at **"dog_app.ipynb"**

## Using Convolutional Neural Networks to Identify Dog Breeds
Thada Jirajaras  
May 31st, 2021

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

Nowaday, AI and computer vision play an important role in detecting objects [1]. There are many
models pretrained with a big dataset for this task. For example, there are models that can identify
objects in the ImageNet dataset which have 1000 classes [2].
Howevers, there are limitations that these pretrained models can identify only the classes that
appear in the trained dataset. For example, in order to specifically identify unseen classes like
some dog breeds, the models need to be newly created. Also, there may be situations where
there are only a few dog images that have the breed labels because it requires a lot of effort to
label images [3]. Thus, the model must use only a few images to train.

### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

With only a few dog images with breed labels, it is challenging to create a model that does not
overfit the training set and give high accuracy on the test set.

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

Accuracy is the acceptable and selected metric because classes are not extremely unbalanced.


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)

1. Human dataset:
(https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
This dataset contains human faces and will be used just for testing the human
detector algorithm. For this project, we will use a face detection algorithm to detect
the human face. As we know that existing face detection algorithms can work pretty
well so this task is not the main focus for this project.
2. Dog dataset:
(https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
This is the main dataset for developing a CNN model to identify dog breeds. The
dataset contains dog images with 133 breed labels. The dataset consist of training
(6680 images), validation (835 images), and test (836 images) sets



### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

In order to achieve high accuracy from fitting the model with only a few images, transfer learning
must be used. Also, the chosen base models should be trained on the dataset that contain some
different dog breeds. Fortunately, all base models from torchvision are trained on the ImageNet
dataset which contains some image categories corresponding to dogs [2]. Thus, one of these
base models will be chosen for transfer learning in this project.

```python
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models

### Write data loaders for training, validation, and test sets
use_cuda = 1
loaders_transfer, data_transfer = create_loader()

## Specify model architecture
model_transfer = models.selected_model(pretrained=True)
model_transfer.classifier[6].out_features = len(data_transfer['train'].classes)
for param in model_transfer.features.parameters():
param.requires_grad = False
if use_cuda:
	model_transfer = model_transfer.cuda()
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr = 0.001, momentum = 0.9)

# train the model
model_transfer = train(num_epochs, loaders_transfer, model_transfer, 
                           optimizer_transfer, criterion_transfer, 
                           use_cuda, 'model_transfer.pt', 
                           mode = 'transfer', test_run = test_run_flag,
                           let_update_tranfer_layer_at_epoch = 20,
                          unfreeze_layer = model_transfer.layer4)

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```



### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

There are some works that use the same datasets to create and test the models [4, 5] and give
the similar benchmark results. For example, Maanav Shah created 4 models including VGG-49,
ResNet-50, Inception V3, and Xception [4]. The model performance comparisons of his work are
as follows.

| Model        | Acc  | Training Time in seconds <br />(20 epochs, batch size = 32) |
| ------------ | ---- | ----------------------------------------------------------- |
| VGG-19       | 46%  | 24.16                                                       |
| ResNet-50    | 82%  | 18.41                                                       |
| Inception v3 | 81%  | 31.83                                                       |
| Xception     | 85%  | 49.07                                                       |




## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

Solution is to write an algorithm that accepts a file path to an image and first determines whether
the image contains a human, dog, or neither. Then, if a dog is detected in the image, return the
predicted breed. if a human is detected in the image, return the resembling dog breed. if neither
is detected in the image, provide output that indicates an error.
Main focus is to improve the accuracy of the CNN model that identifies dog breeds. The model
architecture or parameters will be modified iteratively to achieve at least 60% accuracy on the
test set.

```python
def run_app(img_path):
## handle cases for a human face, dog, and neither
img = cv2.imread(img_path)
if dog_detector(img_path):
	breed = predict_breed_transfer(img_path)
	print(f'''Hi, {breed}''')
elif face_detector(img_path):
	breed = predict_breed_transfer(img_path)
	cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(f'''Hello, human, You look like a ... \n {breed}''')
else:
	raise Exception("Neither is detected in the image")
plt.imshow(cv_rgb)
plt.show()
```

```
Epoch: 1 	Training Loss: 2.297953 	Validation Loss: 0.920713 	Validation Accuracy: 78.44%
Epoch: 2 	Training Loss: 0.779877 	Validation Loss: 0.657823 	Validation Accuracy: 82.16%
Epoch: 3 	Training Loss: 0.562465 	Validation Loss: 0.634714 	Validation Accuracy: 80.84%
Epoch: 4 	Training Loss: 0.461960 	Validation Loss: 0.553500 	Validation Accuracy: 82.99%
Epoch: 5 	Training Loss: 0.398167 	Validation Loss: 0.540369 	Validation Accuracy: 83.11%
Epoch: 6 	Training Loss: 0.343145 	Validation Loss: 0.515508 	Validation Accuracy: 83.71%
Epoch: 7 	Training Loss: 0.336196 	Validation Loss: 0.647430 	Validation Accuracy: 81.56%
Epoch: 8 	Training Loss: 0.279562 	Validation Loss: 0.490302 	Validation Accuracy: 83.83%
Epoch: 9 	Training Loss: 0.268762 	Validation Loss: 0.463448 	Validation Accuracy: 84.67%
Epoch: 10 	Training Loss: 0.241893 	Validation Loss: 0.535265 	Validation Accuracy: 82.99%
Epoch: 11 	Training Loss: 0.234088 	Validation Loss: 0.533402 	Validation Accuracy: 83.47%
Epoch: 12 	Training Loss: 0.222696 	Validation Loss: 0.490432 	Validation Accuracy: 83.95%
Epoch: 13 	Training Loss: 0.201634 	Validation Loss: 0.496213 	Validation Accuracy: 84.07%
Epoch: 14 	Training Loss: 0.195713 	Validation Loss: 0.468101 	Validation Accuracy: 85.03%
Epoch: 15 	Training Loss: 0.175688 	Validation Loss: 0.509956 	Validation Accuracy: 83.11%
Epoch: 16 	Training Loss: 0.180977 	Validation Loss: 0.511798 	Validation Accuracy: 83.59%
Epoch: 17 	Training Loss: 0.174031 	Validation Loss: 0.529860 	Validation Accuracy: 84.31%
Epoch: 18 	Training Loss: 0.164736 	Validation Loss: 0.525740 	Validation Accuracy: 84.79%
Epoch: 19 	Training Loss: 0.172417 	Validation Loss: 0.489292 	Validation Accuracy: 85.75%
Enabled more params to be trained
Epoch: 20 	Training Loss: 0.153272 	Validation Loss: 0.491455 	Validation Accuracy: 85.15%
Epoch: 21 	Training Loss: 0.135427 	Validation Loss: 0.506243 	Validation Accuracy: 85.03%
Epoch: 22 	Training Loss: 0.154449 	Validation Loss: 0.488009 	Validation Accuracy: 84.91%
Epoch: 23 	Training Loss: 0.147306 	Validation Loss: 0.501876 	Validation Accuracy: 85.03%
Epoch: 24 	Training Loss: 0.142897 	Validation Loss: 0.500060 	Validation Accuracy: 85.15%
Epoch: 25 	Training Loss: 0.147585 	Validation Loss: 0.498768 	Validation Accuracy: 85.39%
```



### Refinement

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_

```
shuffle = True
```




## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?

```
Test Loss: 0.515817


Test Accuracy: 82% (690/836)
```

| ![image-20210424202911720](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424202911720.png) | ![image-20210424202931443](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424202931443.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210424203002706](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424203002706.png) | ![image-20210424203040139](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424203040139.png) |
| ![image-20210424203019752](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424203019752.png) | ![image-20210424203100187](C:\Users\OOKBee U\AppData\Roaming\Typora\typora-user-images\image-20210424203100187.png) |



### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

## Reference

1. Medium. 2021. Everything You Ever Wanted To Know About Computer Vision. Here’s A Look
Why It’s So Awesome.. [online] Available at:
<https://towardsdatascience.com/everything-you-ever-wanted-to-know-about-computer-vision
-heres-a-look-why-it-s-so-awesome-e8a58dfb641e> [Accessed 20 April 2021].
2. Pytorch.org. 2021. torchvision.models — Torchvision master documentation. [online]
Available at: <https://pytorch.org/vision/stable/models.html> [Accessed 20 April 2021].
3. Anolytics. 2021. Top Data Labeling Challenges Faced by the Data Annotation Companies.
[online] Available at:
<https://www.anolytics.ai/blog/top-data-labeling-challenges-faced-by-annotation-companies/>
[Accessed 20 April 2021].
4. Medium. 2021. Dog Breed Classifier using CNN. [online] Available at:
<https://medium.com/@maanavshah/dog-breed-classifier-using-cnn-f480612ac27a>
[Accessed 20 April 2021].
5. Medium. 2021. Deep Learning: Build a dog detector and breed classifier using CNN. [online]
Available at:
<https://towardsdatascience.com/deep-learning-build-a-dog-detector-and-breed-classifier-usi
ng-cnn-f6ea2e5d954a> [Accessed 20 April 2021].
6. Ayanzadeh, Aydin & Vahidnia, Sahand. (2018). Modified Deep Neural Networks for Dog
Breeds Identification.