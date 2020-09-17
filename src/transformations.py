""" Code adapted from https://github.com/Spijkervet/SimCLR/blob/148d5987c90c70003d6611c5f22d8346d4649dbe/modules/transformations/simclr.py"""

import torchvision

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        trans_list = [
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.RandomResizedCrop(size=size[1:]),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    ]

        if size[0] == 1:
            trans_list.extend([
                        torchvision.transforms.Grayscale(num_output_channels=1),
                        torchvision.transforms.ToTensor(),
                    ])
        else:
            trans_list.extend([
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ])

        self.train_transform = torchvision.transforms.Compose(trans_list)

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=size[1:]),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return [self.train_transform(x), self.train_transform(x)]

