[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)

# IML WS 23 Project

## Topic: On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/17377))

This topic focuses on the application of semi-factuals and counterfactuals to image data. In comparison to tabular data, the non-discrete nature of image data poses challenges in defining well-behaved and plausible counterfactuals. The authors propose a novel method to generate plausible counterfactuals and semi-factuals (PIECE, PlausIble Exceptionality-based Contrastive Explanations) for black-box ML Model. In this paper, the method is applied to a convolutional neural network (CNN) with the following architecture([cnn architecture](https://github.com/automl-classroom/project-we-are-3/tree/main/Model_Training#readme)). All "exceptional" features in a test image, specifically the features of 128 neurons in the second layer of the CNN, are modified to be "normal" from the perspective of the counterfactual class. This process is undertaken to generate plausible counterfactual or semi-factual images.

The authors employed the MNIST Digits and CIFAR-10 datasets to assess the performance of PIECE. The results on the Digits dataset were highly satisfactory, while the outcomes on CIFAR-10 were deemed less than optimal. We attribute this discrepancy to the inherent differences between the Digits and CIFAR-10 datasets. The Digits dataset, characterized by its simplicity, exhibits relatively straightforward and distinguishable structures in the shapes of digits. Conversely, CIFAR-10, featuring diverse object images, poses challenges due to potentially similar shapes, intricate textures, and less distinct boundaries between objects, making the generation of semi-factuals and counterfactuals more challenging. Consequently, we propose employing PIECE to generate explanations for the [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist) dataset. This dataset shares structural similarities with the Digits dataset, comprising 10 categories of wearable items, and consists of 28x28 pixel grayscale images.

## About PIECE

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/PECIE.png)
_Image Source: [On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17377)_

The core of the PIECE algorithm lies in selecting the feature layer $X$ before the CNN output layer as the basis for generating semifactual and counterfactual instances. It computes the probability density function of all latent features in this feature layer $X$ for the all training images. Using this probability density function, it identifies the exceptional latent features of the input image in this feature layer $X$ and replaces them with the corresponding mean values to generate semi-factual and counterfactual images.

## Repo structure

- Explanation (see the discription [here](https://github.com/automl-classroom/project-we-are-3/tree/main/Explanation#readme))
- Model_Training (see the discription [here](https://github.com/automl-classroom/project-we-are-3/tree/main/Model_Training))
- In the MNIST folder. Added collect.ipynb to collect full connectivity layer data for the full dataset. This part of the data will be used to generate the probability density function. Changed some of the code in re_MNIST.ipynb to test it, and used a new way of evaluating the results of the generated images at the end.
- In the Fashion-MNIST folder. Partial changes to F_MNIST.ipynb to accommodate the new dataset. Other parts are the same as MNIST.

## What have we achieved?

1. Re-implemente some Approches, which inlcudes
  - collecting all the latent features from the training-dataset
  - train the DCGAN for MNISTFashion Dataset
  - train the CNN for MNISTFashio Dataset
2. Testing PIECE on the MNIST and Fashion-MNIST dataset.
3. Run an ablation study, Observe changes in the picture by changing the alpha (probability density function threshold)
4. Evaluate the Plausible of a picture by using a new metic (one-Nearest Neighbor Classifier)

## Installations

To install requirements:

```setup
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Download required large files

You can download pretrained files needed here:

- https://drive.google.com/drive/folders/17kPR0fqcyWLE5qyUK3990_bi4ULrMWg3?usp=sharing

Put 'distribution data' in the MNIST/data folder. Put 'pred_features.pickle' in the MNIST/data folder. Put 'generator.pth' in the MNIST/weights folder.
Put 'collected_fashion.pickle' in the F-MNIST/data folder.

## Results

### MNISTDigits Semifactual / Counterfactual Image Change Process

In this example, an image of the digit 8 was misclassified as 3. We obtained 128 latent features $Xi$ by extracting the output from the CNN for this image. Through the PEICE algorithm, we identified 28 Exceptional Features among these latent features. The importance of these features was assessed using xxx and they were subsequently ranked.

The following figure illustrates the changes in the image after modifying the latent feature values. Notably, after altering the third feature vector, the image was correctly classified by the CNN. On the left side of the decision boundary are images where the classification result changed after modifying latent feature values, interpreted as semifactual. On the right side of the decision boundary are counterfactual images. (more details about this Example see [here](Explanation\PIECE_explanantion_on_MNIST_Digits.ipynb))

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/MNISTDigits_example.png)

We can observe that the gap on the left side of the digit 8 is gradually filled in. In the CNN's perception, this aligns more closely with the latent features of the digit 8.

### MNISTFashion Semifactual / Counterfactual Image Change Process

Similarly, we provided an interpretation for an image from MNIST Fashion. The label for this image was a boot, but the CNN misclassified it as a ankel sneaker. Using the PIECE algorithm, we identified 31 exceptional latent features and made alterations to them. The image was correctly classified after modifying the ninth feature. Interestingly, when we changed the thirteenth exceptional feature, the image was again misclassified as a sneaker, but after making another alteration, it was correctly classified once more.(more details about this Example see [here](Explanation\PIECE_explanation_on_MNISTFashion.ipynb))

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/MNISTFashion_example.png)

In the model's perception, boots are expected to have higher ankles and a more elongated shoe body.

In our attempt on the MNIST Fashion dataset, we try to generate semifactual images through the PIECE method, where initially correctly classified images could be transformed to appear as if they belong to a different class. However, challenges arose due to suboptimal performance of the GAN, and the original images of items in the fashion dataset and the changes introduced in the semifactual images are imperceptible to humans.

The examples of images we finally found do not exhibit particularly satisfactory results (see [here](Explanation\PIECE_explanation_on_MNISTFashion_correctTOincrrect.ipynb))

### Ablation study / Select different alpha observations to generate the effect

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/41.png)  
In this section different alpha's were used for testing and it can be seen that as the alpha's increase the original image becomes more and more similar to 2 (the top half gets shorter and the bottom half gets longer). This is due to the fact that more features of the fully-connected layer were changed. the change in threshold did not result in a particularly large number of features needing to be changed, so the three images generated using PIECE do not differ much.

### Evaluations:

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/28.png)

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/sneaker.png)  
It can be seen that PIECE makes more understandable changes compared to the other algorithms.The Min-Edit and C-Min-Edit algorithms are only concerned with changing the predictions of the classifiers by making subtle changes to the images.

![Image](https://github.com/automl-classroom/project-we-are-3/blob/main/imgs/TABEL.png)  
Several different evaluation methods are used here to evaluate the trustworthiness of the generated images:

- MC-Mean: Posterior mean of MC Dropout on the generated counterfactual image (higher is better)
- MC-STD: Posterior standard deviation of MC Dropout on the generated counterfactual (lower is better).
- NN-Dist: Use NN-Dist to calculate the feature similarity between the generated images and the original dataset (lower is better).
- IM1: The lower the value of IM1, the better the model explains the input samples.
- One-Nearest Neighbor Classifier: This is tested by training a classifier that mixes the generated images with the original data, and if the classifier fails to classify successfully, the accuracy should be close to 50%.
