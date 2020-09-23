""" Code adapted from https://github.com/Spijkervet/SimCLR/blob/148d5987c90c70003d6611c5f22d8346d4649dbe/modules/transformations/simclr.py"""

from torchvision import transforms
from torchvision.models import inception_v3

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        trans_list = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size[1:]),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
        ]

        test_trans_list = [
            transforms.ToPILImage(),
            transforms.Resize(size=size[1:]),
        ]

        if size[0] == 1:
            trans_list.extend([
                transforms.Grayscale(num_output_channels=1),
            ])

            test_trans_list.extend([
                transforms.Grayscale(num_output_channels=1),
            ])

        else:
            trans_list.extend([
                transforms.RandomGrayscale(p=0.2),
            ])

        self.train_transform = transforms.Compose((*trans_list, transforms.ToTensor()))

        self.test_transform = transforms.Compose((*test_trans_list, transforms.ToTensor()))

    def __call__(self, x):
        return [self.train_transform(x), self.train_transform(x)]

class TransformInception:

    def __init__(self, in_features, out_features):
        
        self.in_features = in_features
        self.out_features = out_features

        trans_list = [
            transforms.ToPILImage(),
            transforms.Resize(299), 
            transforms.CenterCrop(299), 
            transforms.ToTensor(),
        ]

        self.transform = transforms.Compose(trans_list)

        self.projector = inception_v3(pretrained=True, transform_input=True, aux_logits=False)

    def __call__(self, x):
        
        t = self.transform(x).unsqueeze(0)
        z = self.projector(t)
        
        return [z[:, self.in_features], z[:, self.out_features]]
