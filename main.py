import random

from keras.layers import Dense
from keras.models import Model

from ImageTrainer import ImageTrainer


# Training first model

trainer = ImageTrainer(
    root = '101_ObjectCategories', exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
)
categories = trainer.get_categories()
data = trainer.build(categories)
num_classes = len(categories)
random.shuffle(data)
trainer.split_dataset(data, num_classes)

trainer.build_model(num_classes)
trainer.compile_model()
trainer.train_model()
history = trainer.history
trainer.evaluate_model()

# Training second model

second = ImageTrainer(root = 'Treinamento', exclude = [])
second_categories = second.get_categories()
second_data = second.build(second_categories)
second_num_classes = len(second_categories)
random.shuffle(second_data)
second.split_dataset(second_data, second_num_classes)

second.build_model_small(second_num_classes)
second.compile_model()
second.train_model()
second.evaluate_model()
second_history = second.history
second.evaluate_model()

## ______________________ Transfer learning

# reference to first model's input
inp = trainer.model.inputs

new_classification_layer = Dense(second_num_classes, activation='softmax')
out = new_classification_layer(trainer.model.layers[-2].output)
model_new = Model(inp, out)

for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model_new.summary()

history3 = model_new.fit(
    second.vectors["x_train"],
    second.vectors["y_train"],
    epochs=10,
    batch_size=32,
    validation_data=(
        second.vectors["x_val"], second.vectors["y_val"]
    )
)
loss, accuracy = model_new.evaluate(second.vectors["x_test"], second.vectors["y_test"], verbose=0)

# Validating with a new image
img, x = second.get_image('Treinamento/Moto/download (1).jfif')
probabilities = model_new.predict([x])
print(probabilities)