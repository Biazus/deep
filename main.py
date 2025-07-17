import random

from ImageTrainer import ImageTrainer


# TODO fine tuning here
trainer = ImageTrainer(root = '101_ObjectCategories', exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces'])
categories = trainer.get_categories()
data = trainer.build(categories)
num_classes = len(categories)

random.shuffle(data)
trainer.split_dataset(data, num_classes)

"""
print("Finished loading %d images from %d categories"%(len(data), num_classes))
print("Train / validation / test split: %d, %d, %d"%(len(vectors["x_train"]), len(vectors["x_val"]), len(vectors["x_test"])))
print("Training data shape: ", vectors["x_train"].shape)
print("Training labels shape: ", vectors["y_train"].shape)
"""
trainer.build_model(num_classes)

trainer.compile_model()

trainer.train_model()
history = trainer.history

trainer.evaluate_model()
