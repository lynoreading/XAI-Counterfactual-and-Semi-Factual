## Model Training

In this document, there are three files pertaining to the training of [DCGAN](DCGANs_training_MNISTFashion.ipynb), [CNN](CNN_Traning_MNISTfashion.ipynb), and [finding 50 images examples](get_image_examples_from_MNISTFashion.ipynb) on Colab. The dataset utilized for these tasks is MNIST Fashion.

### DCGAN model

The DCGAN was trained for a total of 70 epochs, with a learning rate of 0.0002 and a batch size of 128. The training dataset consisted of 60,000 images from MNIST Fashion.

#### Generator:

```bash
Generator(
    (main): Sequential(
        (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): ConvTranspose2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(2, 2), bias=False)
        (13): Tanh()
    )
)
```

#### Discriminator

```bash
Discriminator(
    (main): Sequential(
        (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (12): Sigmoid()
    )
)
```

### CNN model

The CNN model was trained for 10 epochs at lr=0.1, 20 epochs more at lr=0.01, 10 epochs more at lr=0.001 and 10 epochs more at lr=0.0002. The accuracy on the test set, comprising 10,000 images, is 85.84% ([evaluation file](https://github.com/automl-classroom/project-we-are-3/blob/main/Explanation/cnn_MNISTFashion_eval.ipynb)).

```bash
CNN(
    (main): Sequential(
        (0): Conv2d(1, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout2d(p=0.1, inplace=False)
        (4): Conv2d(8, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): Dropout2d(p=0.1, inplace=False)
        (8): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): ReLU(inplace=True)
        (11): Dropout2d(p=0.1, inplace=False)
        (12): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (13): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (14): ReLU(inplace=True)
        (15): Dropout2d(p=0.2, inplace=False)
        (16): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (18): ReLU(inplace=True)
    )
    (classifier): Sequential(
        (0): Linear(in_features=128, out_features=10, bias=True)
    )
)
```
