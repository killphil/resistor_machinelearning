from ultralytics import YOLO
import numpy as np
from pathlib import Path
from lib.resistorpreprocessing import detect_rotate
import tensorflow as tf
import models.models as mdl
import matplotlib.pyplot as plt
import time
import cv2
import pandas as pd
from lib.helpfunctions import get_image

# Directory for test data
testdir = './Datasets/Preprocessed_Dataset/test'

# Parameters
num_classes = 127
batch_size = 20
img_height = 256
img_width = 256
img_shape = (img_width, img_height, 3)
input_shape = (None,) + img_shape

# Prepare the test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    testdir,
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    interpolation='lanczos5'
)

# Get a sample image
dataset = './Datasets/Original_Dataset'
image_path = get_image(dataset)
image = cv2.imread(image_path)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
ret,cropped_image = detect_rotate(image)
if(ret==True):
    plt.title('Preprocessed Image')
    plt.imshow(cropped_image)
    cropped_resized = cv2.resize(cropped_image, (256, 256))
    sample_image = np.expand_dims(cropped_resized, axis=0)
else:
    ValueError("Couldn't detect resistor!")



# Collect results in a list
results = []
i=0
for model in mdl.all_models:
    name = mdl.all_models_names[i]
    print(f'\n\n {name}')
    
    # Measure prediction interference time
    time_start = time.time()
    model.predict(sample_image)
    time_end = time.time()
    interference_time = time_end - time_start
    interference_time = f'{interference_time:.4f}'

    # Evaluate the model
    eval_results = model.evaluate(test_ds)
    loss = f'{eval_results[0]:.4f}'
    accuracy = f'{eval_results[1] * 100:.2f}%'
    top_5 = f'{eval_results[2] * 100:.2f}%'

    print(f'\nInterference Time: {interference_time}')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Top-5 Accuracy: {top_5}')

    # Append results to the list
    results.append({
        'Model Name': name,
        'Accuracy': accuracy,
        'Top-5 Accuracy': top_5,
        'Loss': loss,
        'Interference Time [s]': interference_time
    })

    i += 1

# Convert results to a DataFrame
results_df = pd.DataFrame(results)


# Save the results to an Excel file
output_path = Path('./models/results.xlsx')
output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
results_df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")