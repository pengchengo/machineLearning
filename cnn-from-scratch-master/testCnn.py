import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from PIL import Image

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1]
train_labels = mnist.train_labels()[:1]
print(train_images[0].shape) 
#print('train_images:', train_images)
print('train_labels:', train_labels)

#pil_img = Image.fromarray(np.uint8(train_images[0]))
#pil_img.show()

testShape = np.zeros((1, 1, 2))
#print('testShape:', testShape)
#print('testShape.shape:', testShape.shape)

image = train_images[0]
conv = Conv3x3(8)
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

out = conv.forward((image / 255) - 0.5)
out = pool.forward(out)
out = softmax.forward(out)

label = train_labels[0]
gradient = np.zeros(10)
gradient[label] = -1 / out[label]

lr=.005
# Backprop
gradient = softmax.backprop(gradient, lr)
#gradient = pool.backprop(gradient)
#gradient = conv.backprop(gradient, lr)