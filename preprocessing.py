import cv2      # OpenCV library for computer vision
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os

""" Loading the Data """
# Read and store list of file names
def read_image_filenames(path):
    extensions = ["*.tif", "*.tiff", "*.jpeg", "*.jpg", "*.png"]
    image_filenames = []
    for ext in extensions:
        image_filenames.extend(glob(os.path.join(path, ext)))
    return image_filenames

# Read the image as an array
def read_image(image_filename):
    image = cv2.imread(image_filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to display an image
def display_image(image, only_a_path):
  if only_a_path:
    img = cv2.imread(image) # reads image in BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converts BGR to RGB
  else:
    img = image
  plt.axis('off')
  plt.imshow(img)
  plt.show()

""" Pre-processing """
# Helper method to classify pixel
def is_black(pixel):
  return pixel[0]<=10 and pixel[1]<=10 and pixel[2]<=10
def is_red(pixel):
  if pixel[0] > 0:
    return pixel[1]/pixel[0]<=1 and pixel[2]/pixel[0]<=1
  else:
    return False
def is_yellow(pixel):
  if pixel[1] > 0:
    return pixel[0] / pixel[1] >= 0.5 and pixel[2] / pixel[1] < 0.2
  else:
    return False
def is_green(pixel):
  if pixel[1] > 0:
    return pixel[0] / pixel[1] < 0.5 and pixel[2] / pixel[1] < 0.5
  else:
    return False
def is_blue(pixel):
  return not (is_black(pixel) or is_red(pixel) or is_yellow(pixel) or is_green(pixel))

# Count amount of red, yellow, green, and blue pixels
def count_pixels(img):
  red_count = 0
  yellow_count = 0
  green_count = 0
  blue_count = 0
  for row in img:
    for pixel in row:
      if is_red(pixel):
        red_count+=1
      elif is_yellow(pixel):
        yellow_count+=1
      elif is_green(pixel):
        green_count+=1
      elif is_blue(pixel):
        blue_count+=1
  return red_count, yellow_count, green_count, blue_count

def pre_process(img_filename):
  # Print file name you are working on
  print(img_filename)
  # Read image
  plume = read_image(img_filename)
  # Display image after isolating the plume
  #display_image(plume, False)
  # Count the relevant pixels and return
  count = count_pixels(plume)
  print("(R, Y, G, B): ",count)
  return count

""" Test """
# Test images
test_filenames = ["C://Users//gabri//Documents//MethaneDatathon//Data//images//ang20160910t185702_S00006_r2533_c411_ctr_rotated.tif"]
for i in range(1):
  pre_process(test_filenames[i])

""" Pre-process training images """
# File locations
ROOT_PATH = "C://Users//gabri//Documents//MethaneDatathon//"
DATA_ROOT_PATH = ROOT_PATH + "Data//"
DATA_CSV = "combined_data.csv"

# Open permian data file
data_df = pd.read_csv(DATA_ROOT_PATH+DATA_CSV)

# Drop unnecessary columns
names = ['source_id', 'source_type', 'plume_id', 'plume_lat', 'plume_lon', 
         'datetime', 'ipcc_sector', 'rgb_tif']
for name in names:
  data_df = data_df.drop(name, axis = 1)

# Display info
print(data_df.info())
print(data_df.iloc[0])

# Add engineered features
red_pixels = []
yellow_pixels = []
green_pixels = []
blue_pixels = []
total = len(data_df.axes[0])
current = 0
for index, row in data_df.iterrows():
  current += 1
  print("Processing: ", current,"/",total)
  pixel_count = pre_process(DATA_ROOT_PATH + row['plume_tif'])
  red_pixels.append(pixel_count[0])
  yellow_pixels.append(pixel_count[1])
  green_pixels.append(pixel_count[2])
  blue_pixels.append(pixel_count[3])

# Add filenames to dataframe
data_df['red_pixels'] = red_pixels
data_df['yellow_pixels'] = yellow_pixels
data_df['green_pixels'] = green_pixels
data_df['blue_pixels'] = blue_pixels

# Most images have a white dot that gets labelled as red, remove it
data_df['red_pixels'] -= 1

# Save counts to a csv
data_df.to_csv(ROOT_PATH+'data_pixel_counts.csv', index=False)







""" Pre-process test images """
TEST_ROOT_PATH = DATA_ROOT_PATH + "TestData//"
# Make copy of test filenames
test_filenames = [
    "1A_ctr_rotated.tif",
    "2A_ctr_rotated.tif",
    "3A_ctr_rotated.tif",
    "4A_ctr_rotated.tif",
    "5A_ctr_rotated.tif",
    "6A_ctr_rotated.tif",
    "7A_ctr_rotated.tif",
    "8A_ctr_rotated.tif",
    "9A_ctr_rotated.tif",
    "10A_ctr_rotated.tif",
    "11A_ctr.tif",
    "12A_ctr.tif",
    "13A_ctr.tif",
    "14A_ctr.tif",
    "15A_ctr.tif",
    "16A_ctr.tif",
    "17A_ctr.tif",
    "18A_ctr.tif",
    "19A_ctr.tif",
    "20A_ctr.tif"
]

# Add filenames to dataframe
test_df = pd.DataFrame(test_filenames, columns =['filename'])

# Display info
print(test_df.info())
print(test_df.iloc[0])

# Add engineered features
red_pixels = []
yellow_pixels = []
green_pixels = []
blue_pixels = []
total = len(test_df.axes[0])
current = 0
for index, row in test_df.iterrows():
  current += 1
  print("Processing ", current,"/",total)
  pixel_count = pre_process(TEST_ROOT_PATH + row['filename'])
  red_pixels.append(pixel_count[0])
  yellow_pixels.append(pixel_count[1])
  green_pixels.append(pixel_count[2])
  blue_pixels.append(pixel_count[3])

# Add filenames to dataframe
test_df['red_pixels'] = red_pixels
test_df['yellow_pixels'] = yellow_pixels
test_df['green_pixels'] = green_pixels
test_df['blue_pixels'] = blue_pixels

# Most images have a white dot that gets labelled as red, remove it
test_df['red_pixels'] -= 1

test_df.to_csv(ROOT_PATH + 'test_pixel_count.csv', index=False)

