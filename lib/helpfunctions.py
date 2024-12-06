import os
import random

def get_image(folder_path):
    # Get the list of class directories
    classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    # Randomly select a class
    selected_class = random.choice(classes)
    class_path = os.path.join(folder_path, selected_class)
    # Get a list of images in the selected class
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    # Randomly select an image
    selected_image = random.choice(images)
    # Return the full path of the image and the class name
    return os.path.join(class_path, selected_image)

def get_image_and_class(folder_path):
    # Get the list of class directories
    classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    # Randomly select a class
    selected_class = random.choice(classes)
    class_path = os.path.join(folder_path, selected_class)
    # Get a list of images in the selected class
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    # Randomly select an image
    selected_image = random.choice(images)
    # Return the full path of the image and the class name
    return os.path.join(class_path, selected_image), selected_class


def get_class_names_from_folder(folder_path):
    # List all subdirectories in the dataset folder
    class_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    return sorted(class_names)  # Sort for consistency with training order


def get_predicted_class(classnames,prediction):
    return classnames[prediction]

