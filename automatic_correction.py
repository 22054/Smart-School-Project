import student_dataset.load_image_dataset as ocr_data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
#warnings.filterwarnings('ignore')

# Paths to your files
images_file = "student_dataset/image_data/train-images-idx3-ubyte"
labels_file = "student_dataset/image_data/train-labels-idx1-ubyte"

# Create dataset
dataset = ocr_data.MyDataset(images_file, labels_file)

# Show 10 samples starting at index 0
#dataset.show_sample(idx=0, count=10)

#print('1', dataset.images[0])
print('2', dataset.char_labels[:10])
print('3', dataset.labels[:10])
print('4', dataset[:10])