import torch
import matplotlib.pyplot as plt
from src.DataAugmentation.DataAugmentation import AugmentationScheme as AS
import numpy as np


# augments batches of images
def augment_batch(images, labels):

    # Create a new array that contains the batch of images that have been reformatted to accommodate the aug lib
    reformatted_images_list = []
    for i in images:
        reformatted_images_list.append(i[0])
    reformatted_images = np.asarray(reformatted_images_list)

    # Create instance of DA class in order to create desired augmentation scene
    augSc = AS(reformatted_images, labels)
    # Choose desired augmentations
    augSc.add_augmentation("Flip_lr", percent=1)
    augSc.add_augmentation("Rotate", lo=-45, hi=45)
    # Create augmented batch of images and labels
    augmented_batch, aug_labels = augSc.augment_images()

    # Displays original image + augmented image (for testing)
    display_image(reformatted_images[0])
    display_image(augmented_batch[0])

    # Reformat augmented batch into the shape that the  rest of the code wants
    augmented_batch = augmented_batch.reshape(64, 1, 28, 28)

    # Convert images stored in numpy arrays to tensors
    t_augmented_images = torch.from_numpy(augmented_batch)
    t_labels = torch.from_numpy(labels)

    return t_augmented_images, t_labels


def display_image(image):

    # Image must be numpy array
    plt.imshow(image, cmap='gray')
    plt.show()
