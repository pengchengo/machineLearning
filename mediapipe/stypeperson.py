import numpy as np
import mediapipe as mp
import cv2
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

IMAGE_FILENAMES = ['mz.png']

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("new",img)
  cv2.waitKey(0)

# Create the options that will be used for FaceStylizer
#base_options = python.BaseOptions(model_asset_path='face_stylizer.task')
base_options = python.BaseOptions(model_asset_path='face_stylizer_oil_painting.task')
options = vision.FaceStylizerOptions(base_options=base_options)

# Create the face stylizer
with vision.FaceStylizer.create_from_options(options) as stylizer:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be stylized
    image = mp.Image.create_from_file(image_file_name)
    # Retrieve the stylized image
    stylized_image = stylizer.stylize(image)

    # Show the stylized image
    rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
    resize_and_show(rgb_stylized_image)