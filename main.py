import os

import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model


root = '101_ObjectCategories'
exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
train_split, val_split = 0.7, 0.15



def get_categories(root, exclude):
	# E.g. ['101_ObjectCategories/cat', '101_ObjectCategories/dog', ...]
	categories = [x[0] for x in os.walk(root) if x[0]][1:]
	return [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

def get_image(path):
	# helper function to load image and return it as input vector
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # add batch dimension to array ( e.g. [1, 255, 255, 3])
    x = preprocess_input(x)
    return img, x

def build(categories):
# Load all the images from root folder
	data = []
	for c, category in enumerate(categories):
	    images = [os.path.join(dp, f) for dp, dn, filenames
	              in os.walk(category) for f in filenames
	              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
	    for img_path in images:
	        img, x = get_image(img_path)
	        data.append({'x':np.array(x[0]), 'y':c})
	return data

def split_dataset(data, num_classes, training=0.7, validation=0.15, test=0.15):
	if sum([training, validation, test]) != 1:
		raise ValueError("Sum of subsets should be 1")

	idx_val = int(train_split * len(data))
	idx_test = int((train_split + val_split) * len(data))
	train = data[:idx_val]
	val = data[idx_val:idx_test]
	test = data[idx_test:]

	x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
	x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
	x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]

	# Converts to float32 and normalize between 0 and 1 with float division by 255
	x_train = x_train.astype('float32') / 255.
	x_val = x_val.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.

	# convert labels to one-hot vectors (e.g. [0, 1, 2] would convert to [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]] )
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return {
		"x_train": x_train,
		"y_train": y_train,
		"x_val": x_val,
		"y_val": y_val,
		"x_test": x_test,
		"y_test": y_test
	}
# build the network
def build_model(x_train):
	model = Sequential()
	print("Input dimensions: ",x_train.shape[1:])

	model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.25))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	model.summary()
	return model

categories = get_categories(root, exclude)
data = build(categories)
num_classes = len(categories)

random.shuffle(data)
vectors = split_dataset(data, num_classes)


# summary
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(vectors["x_train"]), len(vectors["x_val"]), len(vectors["x_test"])))
print("training data shape: ", vectors["x_train"].shape)
print("training labels shape: ", vectors["y_train"].shape)

"""
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)
"""
model = build_model(vectors["x_train"])

# compile the model to use categorical cross-entropy loss function and adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(vectors["x_train"], vectors["y_train"],
                    batch_size=16,
                    epochs=10,
                    validation_data=(vectors["x_val"], vectors["y_val"]))

"""
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()
"""
loss, accuracy = model.evaluate(vectors["x_test"], vectors["y_test"], verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


## VGG16

vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()