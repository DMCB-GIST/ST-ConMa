import torchvision.transforms as transforms
import random
from PIL import ImageFilter

class GaussianBlur:
    def __init__(self, sigma: list = [0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x): 
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

def get_augmentation_transforms(augmentation_type: str):
    """
    Get augmentation transforms based on type.

    Args:
        augmentation_type: one of 'aug1', 'aug2', 'aug3'
            - aug1: unaugmented view (used for IGC with gene) - no flips for better gene alignment
            - aug2: augmented view 1 (used for IIC with aug2)
            - aug3: augmented view 2 (used for IIC with aug3)
    Returns:
        composed transform
    """

    # Build transform lists
    aug1_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]

    aug2_list = [
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]

    aug3_list = [
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]

    base_transforms = {
        "aug1": transforms.Compose(aug1_list),
        "aug2": transforms.Compose(aug2_list),
        "aug3": transforms.Compose(aug3_list)
    }

    return base_transforms[augmentation_type]

def get_train_augmentation():

    return (
        get_augmentation_transforms("aug1"),
        get_augmentation_transforms("aug2"),
        get_augmentation_transforms("aug3")
    )