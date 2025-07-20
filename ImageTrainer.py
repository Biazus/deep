import os
from typing import Any

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical


class ImageTrainer:
    root = None
    exclude = None
    vectors = None
    model = None
    history = None

    def __init__(self, root, exclude, *args, **kwargs):
        self.root = root
        self.exclude = exclude

    def get_categories(self):
        """
        Returns a list of categories.
        :return:
        """
        # E.g. ['101_ObjectCategories/cat', '101_ObjectCategories/dog', ...]
        categories = [x[0] for x in os.walk(self.root) if x[0]][1:]
        return [c for c in categories if c not in [os.path.join(self.root, e) for e in self.exclude]]

    def get_image(self, path):
        """
        helper function to load image and return it as input vector
        :param path:
        :return:
        """
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # add batch dimension to array ( e.g. [1, 255, 255, 3])
        x = preprocess_input(x)
        return img, x

    def build(self, categories):
        """
        Load all the images from root folder and build a list of dicts containing the img vector and the label
        :param categories:
        :return:
        """
        data = []
        for c, category in enumerate(categories):
            images = [os.path.join(dp, f) for dp, dn, filenames
                      in os.walk(category) for f in filenames
                      if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg', '.jfif']]
            for img_path in images:
                img, x = self.get_image(img_path)
                data.append({'x':np.array(x[0]), 'y':c})
        return data

    def split_dataset(
            self,
            data: list,
            num_classes: int,
            training: float = 0.7,
            validation: float = 0.15,
            test: float = 0.15
    ) -> None:
        """
        Split and normalize data into training, validation and test sets
        :param data:
        :param num_classes:
        :param training:
        :param validation:
        :param test:
        :return:
        """
        if sum([training, validation, test]) != 1:
            raise ValueError("Sum of subsets should be 1")

        idx_val = int(training * len(data))
        idx_test = int((training + validation) * len(data))
        train = data[:idx_val]
        val = data[idx_val:idx_test]
        test = data[idx_test:]

        x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
        x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
        x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]

        # Convert to float32 and normalize between 0 and 1 with float division by 255
        x_train = x_train.astype('float32') / 255.
        x_val = x_val.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # convert labels to one-hot vectors (e.g. [0, 1, 2] would convert to [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]] )
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        self.vectors =  {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test
        }

    def build_model_small(self, num_classes: int) -> Any:
        """
        A smaller model
        :return:
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=self.vectors["x_train"].shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # deeper layers
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(0.25))

        # create vector of features from last conv
        self.model.add(Flatten())

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        # avoid strong neurons to dominate results, reducing overfitting
        self.model.add(Dropout(0.5))

        # last layer. Default is Dense with softmax
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def build_model(self, num_classes: int) -> Any:
        """
        Create the network
        :param num_classes:
        :return:
        """
        # build the network
        self.model = Sequential()
        print("Input dimensions: ",self.vectors["x_train"].shape[1:])
        # First layer. 32 filters on 3x3 dimension works well for most images
        self.model.add(Conv2D(32, (3, 3), input_shape=self.vectors["x_train"].shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # deeper layers
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # enforces network to not rely on specific filters, reducing overfitting
        self.model.add(Dropout(0.25))

        # create vector of features from last conv
        self.model.add(Flatten())

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        # avoid strong neurons to dominate results, reducing overfitting
        self.model.add(Dropout(0.5))

        # last layer. Default is Dense with softmax
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def compile_model(self):
        """
        Compile the model to use categorical cross-entropy loss function and ada delta optimizer
        :return:
        """
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train_model(self):
        self.history = self.model.fit(
            self.vectors["x_train"], self.vectors["y_train"],
            batch_size=16,
            epochs=10,
            validation_data=(self.vectors["x_val"], self.vectors["y_val"])
        )

    def evaluate_model(self):
        """
        Evaluate the model accuracy and loss
        :return:
        """
        loss, accuracy = self.model.evaluate(self.vectors["x_test"], self.vectors["y_test"], verbose=0)
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)

    def transfer_learning(self, network)->None:
        network.layers[0].set_weights(self.model.get_layer('fc2').get_weights())
