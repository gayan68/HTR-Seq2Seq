import albumentations as A


def aug_transforms(img, aug_prob=0.5):
    # albumentations transforms for text augmentation 
    aug = A.Compose([

        # geometric augmentation
        A.Affine(rotate=(-1, 1), shear={'x':(-30, 30), 'y' : (-5, 5)}, scale=(0.6, 1.2), translate_percent=0.02, mode=1, p=aug_prob),

        # perspective transform
        #A.Perspective(scale=(0.05, 0.1), p=0.5),

        # distortions
        A.OneOf([
            A.GridDistortion(distort_limit=(-.1, .1), p=aug_prob),
            A.ElasticTransform(alpha=60, sigma=20, alpha_affine=0.5, p=aug_prob),
        ], p=0.5),

        # erosion & dilation
        A.OneOf([
            A.Morphological(p=aug_prob, scale=3, operation='dilation'),
            A.Morphological(p=aug_prob, scale=3, operation='erosion'),
        ], p=0.5),

        # color invertion - negative
        #A.InvertImg(p=0.5),

        # color augmentation - only grayscale images
        A.RandomBrightnessContrast(p=aug_prob, brightness_limit=0.2, contrast_limit=0.2),
        
        # color contrast
        A.RandomGamma(p=aug_prob, gamma_limit=(80, 120)),
    ])

    img  = aug(image = img)['image']
    img = 255 - img
    return img
