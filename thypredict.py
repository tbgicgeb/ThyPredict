#!/usr/bin/env python
# coding: utf-8

import image_slicer
import numpy as np
import random
import os
import sys
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import PngImagePlugin
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from os import listdir
import cv2
from glob import glob
from utils import calculate_image_percentages, calculate_ptc_class_percentages, calculate_class_2n_3e_4i_percentages #Import the function from utils.py
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
import subprocess

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # limit memory to be allocated
#K.tensorflow_backend.set_session(tf.Session(config=config)) # create sess w/ above settings
sess = tf.compat.v1.Session(config=config)

#check for Arg pass
if len(sys.argv) != 2:
    print ("Usage: python thypridict.py <image_name>")
    sys.exit(1)

# Get the image name from the command-line argument
image_name = sys.argv[1]

# Update the path variable with the directory and the provided image name
path = os.path.join('input-image', image_name)  # Adjust as necessary for your directory structure


# List of directories to check and delete if they exist
directories_to_check = [
    'main-crop', 
    'stageI-prediction', 
    'stageI-sliced', 
    'thyroid-pipe_copy', 
    'stageII-slice_350', 
    'stageII-prediction',
    '_in_for_stage-III'
]

# Check for existing directories
existing_dirs = [d for d in directories_to_check if os.path.exists(d)]

if existing_dirs:
    print("The following directories already exist and contain previous results:")
    for directory in existing_dirs:
        print(f"- {directory}")
    
    # Prompt user for confirmation
    response = input("Would you like to delete these directories and continue? (yes/no): ").strip().lower()
    if response == 'yes':
        for directory in existing_dirs:
            shutil.rmtree(directory)  # Recursively delete the directory
            print(f"Deleted directory: {directory}")
    else:
        print("Please save your results and run the program again when ready.")
        sys.exit(1)  # Exit the script if user does not want to continue

output_dir = 'stageI-sliced'
#rm -rf main-crop  stageI-prediction slice_350 stageI-sliced thyroid-pipe_copy stageII-slice_350 stageII-prediction _in_for_stage-III

# Ensure the output directory exists, if not create it
os.makedirs(output_dir, exist_ok=True)


# Check the file format and convert to TIFF if not already in TIFF format
file_name, file_extension = os.path.splitext(path)
if file_extension.lower() != '.tiff':
    # Open the image and convert to TIFF
    with Image.open(path) as img:
        tiff_path = f"{file_name}.tiff"
        img.save(tiff_path, format='TIFF')
        path = tiff_path  # Update path to the new TIFF file

# Open the image to get its dimensions
with Image.open(path) as img:
    width, height = img.size

# Calculate the number of tiles using the formula
tile_size = 1000  # Tile size in pixels
num_tiles_width = -(-width // tile_size)  # Equivalent to math.ceil(width / tile_size)
num_tiles_height = -(-height // tile_size)  # Equivalent to math.ceil(height / tile_size)
num_tiles = num_tiles_width * num_tiles_height

print(f"Slicing into {num_tiles} tiles")

####no of tiles 
im_new= image_slicer.slice(path, num_tiles, save=False)   ## Dynamically calculated tile number
image_slicer.save_tiles(im_new, prefix='img',format='tiff', directory=output_dir)

##cell 2
classifier = load_model('models/Stage-I.h5', compile=False)

# Define the directory containing images
image_dir = 'stageI-sliced'

# Get a list of all files in the directory
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tiff')]

# Loop through all image files
for path in image_paths:
    img1 = load_img(path, target_size=(299, 299))  # Load and resize the image
    plt.imshow(img1), plt.show()  # Optionally display the image

    img = img_to_array(img1)  # Convert the image to a NumPy array
    img = img / 255  # Normalize the image data
    img = np.expand_dims(img, axis=0)  # Add a batch dimension

    prediction = classifier.predict(img, batch_size=None, steps=1)  # Make predictions
    print(f"Prediction for {path}: {prediction}")  # Print the prediction


### cell-3

# Set parameters
batch_size = 32
#evaluation_data_dir = 'thyroid-pipe_copy' # where the sliced image stored till parent dir
# Get the current file's directory
evaluation_data_dir = os.path.dirname(os.path.abspath(__file__))
predict1 = 'stageI-prediction'  # Set this to your desired output directory

# Create directories for each class if they don't already exist
class_labels = ['stageI-sliced'] #this lable directory will used to extract sliced image for further analysis (like resize and catagorization)


### cell-4

predicted_class_labels = ['FND', 'Discard', 'Follicles', 'Papillae']  # Change these to your class names

# Create directories for each predicted class if they don't already exist
for label in predicted_class_labels:
    os.makedirs(os.path.join(predict1, label), exist_ok=True)

# Set up the data generator
evaluation_datagen = ImageDataGenerator(rescale=1./255)
evaluation_generator = evaluation_datagen.flow_from_directory(
    evaluation_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    classes=class_labels,
    class_mode='categorical',
    shuffle=False
)
# Get true classes and class labels
true_classes = evaluation_generator.classes
class_indices = list(evaluation_generator.class_indices)

### cell-5

# Make predictions
Y_pred = classifier.predict(evaluation_generator, steps=(1025 // batch_size) + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Print confusion matrix and classification report
print('Confusion Matrix')
cm = metrics.confusion_matrix(true_classes, y_pred)
print(cm)
print('Classification Report')
print(metrics.classification_report(true_classes, y_pred))

# Save images to output folder based on predicted class
for i in range(len(evaluation_generator.filenames)):
    img_path = os.path.join(evaluation_data_dir, evaluation_generator.filenames[i])
    pred_class = predicted_class_labels[y_pred[i]]
    output_path = os.path.join(predict1, pred_class, os.path.basename(img_path))
    shutil.copy(img_path, output_path)

### cell-6

# In[2]:
# Function to crop image to the center
def crop_center(pil_img,crop_width,crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width-crop_width)//2,
                            (img_height-crop_height)//2,
                            (img_width+crop_width)//2,
                            (img_height+crop_height)//2))


# In[3]:

################### Add loop for all ['Adenoma', 'Discard', 'Follicles', 'Papillae'] classes
# Loop over each folder inside base_dir

for class_name in os.listdir(predict1):
  class_dir = os.path.join(predict1, class_name)
  #print (class_dir)


# In[4]:
# Below code perform center cropping on images in the stageI-prediction directory and save the cropped image in the main-crop directory
save_dir = 'main-crop'
#os.makedirs(save_dir)

for images in os.listdir(predict1):
    #print(images)
    os.makedirs(save_dir+'/'+images, exist_ok=True)
    for i in os.listdir(predict1+'/'+ images):
      #print(i)
     # check if the image ends with .tiff
      if (i.endswith(".tiff")):
            img_name=i
            ##print(img_name)
            im = Image.open(predict1+'/'+ images+'/'+i)
            x,y = im.size
            #print(x,y)

            r1= x%350
            if r1 <=175:
                r1=x-r1
                #print(r1)
            else:
                r1=x+(350-r1)
                #print(r1)
            r2= y%350
            if r2 <=175:
                r2=y-r2
                #print(r2)
            else:
                r2=y+(350-r2)
                #print(r2)

            #center Crop
            im_new = crop_center(im,r1, r2)
            im= im_new.save(f'{save_dir}/' +images + '/' + img_name, quality =100)

print ("The final Croped images of StageI-Prediction are under main-croped folder")


#stageII: 
# This code checks for images in the 'main-crop/Follicles' directory and proceeds to slice them into tiles(350*350) if images are found, if image not found Program halted .

#main_crop_dir = 'main-crop/Follicles'
sliced2 = 'stageII-slice_350'

follicles_dir = os.path.join(save_dir, 'Follicles') #save_dir is variable of main-crop folder which contains all the final-prediction of stageI
#print (f"{follicles_dir}")

# List image files in the 'Follicles' directory

image_extensions = ('.jpg', '.jpeg', '.png', '.tiff')  # Add other image extensions if needed
images_in_follicles = [f for f in os.listdir(follicles_dir) if f.lower().endswith(image_extensions)]


# Check if main-crop/Follicles directory contains images
if not images_in_follicles:
    calculate_image_percentages(save_dir)
    #raise FileNotFoundError("No images detected in the 'Follicles' class, Program halted.")
    print("No images detected in the 'Follicles' class")
    sys.exit(1)
    

else:
    print(f"Found {len(images_in_follicles)} images in 'Follicles'.")
    calculate_image_percentages(save_dir)
# Loop over the images in 'Follicles' if not empty
# Process each image in main-crop/Follicles
for img_name in images_in_follicles:
    img_path = os.path.join(follicles_dir, img_name)
    print(f"Processing image: {img_path}")

    # Open the image to slice
    im = Image.open(img_path)
    x, y = im.size
    r1 = x // 350
    r2 = y // 350
    print(f"Slicing into {r1} columns and {r2} rows.")

    # Slice and save tiles
    slice_class_dir = os.path.join(sliced2)
    os.makedirs(slice_class_dir, exist_ok=True)
    tiles = image_slicer.slice(img_path, row=r2, col=r1, save=False)
    image_slicer.save_tiles(tiles, format='tiff', directory=slice_class_dir)

# Check if the directory contains any images
if os.listdir(sliced2):  # Checks if the directory is not empty
    print(f"Stage-II Slicing completed for images in 'main-crop/Follicles' and saved to {sliced2}.")
else:
    print("Stage-II slicing failed.")

### cell-8
classifier2 = load_model('models/Stage-II.h5', compile=False) 
predict2 = 'stageII-prediction'  # Set this to your desired output directory

# Create directories for each predicted class if they don't already exist
predicted_class_labels = ['Non-PTC-like_nuclear_fea', 'PTC-like_nuclear_fea']
for label in predicted_class_labels:
    os.makedirs(os.path.join(predict2, label), exist_ok=True)

# Get a list of all image files in the flat directory
image_paths = [os.path.join(sliced2, f) for f in os.listdir(sliced2) if f.endswith('.tiff')]

# Prepare image data generator for preprocessing
evaluation_datagen = ImageDataGenerator(rescale=1./255)

# Preprocess and predict each image
for img_path in image_paths:
    # Load and preprocess the image
    img1 = load_img(img_path, target_size=(299, 299))  # Resize to model's expected input size
    img = img_to_array(img1)  # Convert to NumPy array
    img = img / 255.0  # Normalize the image data
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = classifier2.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = predicted_class_labels[predicted_class_index]

    print(f"Prediction for {img_path}: {predicted_label}")

    # Move the image to the corresponding class directory
    output_path = os.path.join(predict2, predicted_label, os.path.basename(img_path))
    shutil.copy(img_path, output_path)

print("Classification and organization of images completed.")

########################### Display Result-2 Table-  #start

images_exist = False
# Check each ptc & non_ptc class directory for images
for class_name in predicted_class_labels:
    class_dir = os.path.join(predict2, class_name)
    if os.path.isdir(class_dir):
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
        if image_files:  # Check if list is not empty
            images_exist = True
            break

# Conditional logic based on presence of images
if images_exist:
    calculate_ptc_class_percentages(predict2)
else:
    print("Error: No images predicted in PTC_Like or Non-PTC_like classes.")
    sys.exit(1)

 ######################  Display Result-2 Table- #End   

# Optional: Calculate evaluation metrics if you have true labels
# Assuming you have a list of true labels corresponding to the images
# true_classes = [...]  # Define your true labels here
# y_pred = [np.argmax(classifier2.predict(np.expand_dims(img_to_array(load_img(p)), axis=0)), axis=1)[0] for p in image_paths]

# Print confusion matrix and classification report (if true labels are available)
# print('Confusion Matrix')
# cm = metrics.confusion_matrix(true_classes, y_pred)
# print(cm)
# print('Classification Report')
# print(metrics.classification_report(true_classes, y_pred))

# Dr.Shweta code

# Set image dimensions
IMG_WIDTH = 256  # Width of the input images
IMG_HEIGHT = 256  # Height of the input images
IMG_CHANNELS = 3  # Number of channels (3 for RGB)

# Function to read and preprocess images
def readImage(path):
    img = Image.open(path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img, dtype=np.uint8)
    img = img / 255.
    return img

# Load the model
seg_model = tf.keras.models.load_model('models/preprocessing-roi.keras',safe_mode=False)

# ROI Extraction from Images

# Extracts the Region of Interest (ROI) from an image using a pre-trained segmentation model.
def ROI_extraction(image_path):

   # Read the input image
    image = readImage(image_path)
    if image is None:
        print("Image not loaded, please check the path.")
        return None

    img = np.expand_dims(image, axis=0)
    pred = seg_model.predict(img)*255
    mask = np.where(pred[0] > 128, 255, 0).astype(np.uint8)


    mask_final = cv2.merge([mask, mask, mask])

    # Apply the mask to the original image to extract the ROI
    result = cv2.bitwise_and(image, image, mask=mask_final[:,:,0])
    res = np.array(Image.fromarray(np.uint8(result*255)).resize((350,350)))
    name = os.path.splitext(os.path.basename(image_path))[0]
    # enter the output directory below
    stage2_result = '_in_for_stage-III'
    os.makedirs(os.path.join(stage2_result, label), exist_ok=True)
    cv2.imwrite(os.path.join(stage2_result, str(name+'.tiff')),res[: , : , ::-1])
    return res


# Create a dataset of extracted ROIs from a directory of images.
def create_dataset(path):
    file_paths = glob(path)
    result = np.array(list(map(ROI_extraction, file_paths)))
    return result
#dataset = create_dataset("/content/drive/MyDrive/modclass5cORIGINAL/*")
dataset = create_dataset("stageII-prediction/PTC-like_nuclear_fea/*")


######run Matlab.py 

# Define the output file path
output_file = 'stage-III-processing_output.txt'

#check  if the file exists and delete it
if os.path.exists(output_file):
    os.remove(output_file)
    #print(f"{output_file} has been deleted.")

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Define the path to the matlab.py script
    matlab_script = 'matlab.py'

    # Execute the matlab.py script and redirect output to the file
    try:
        subprocess.run(['python', matlab_script], stdout=f, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print(f"An error occurred while running matlab.py: {e}")


# open the file stage-III-processing_output.txt and calculate the heighest value belonging class and store it to their class directory 

# Directory to store results
final_result_dir = 'final_result'
os.makedirs(final_result_dir, exist_ok=True)

# Class folders
classes = {'class2n': [], 'class3e': [], 'class4i': []}

# Read the output file
with open('stage-III-processing_output.txt', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines)):
        if "Processing image:" in lines[i]:
            image_line = lines[i].strip()
            image_path = image_line.split(': ')[-1]

        # Look for the line that contains exactly three numeric values
        if any(char.isdigit() for char in lines[i]):
            try:
                values_line = lines[i].strip()
                values = [float(v) for v in values_line.split() if v.replace('.', '', 1).isdigit()]
                if len(values) == 3:
                    # Determine the class with the highest value
                    max_index = values.index(max(values))
                    class_name = ['class2n', 'class3e', 'class4i'][max_index]
                    classes[class_name].append(image_path)
            except ValueError:
                continue

# Create class directories and copy images
for class_name, images in classes.items():
    class_dir = os.path.join(final_result_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    for image_path in images:
        if os.path.exists(image_path):
            shutil.copy(image_path, class_dir)

print("Images have been categorized and copied based on the highest class value.")


print("Final Results:")
print("Stage-I Result")
calculate_image_percentages(save_dir)
print("Stage-II Result")
calculate_ptc_class_percentages(predict2)
print("Stage-III Result")
calculate_class_2n_3e_4i_percentages(final_result_dir)
