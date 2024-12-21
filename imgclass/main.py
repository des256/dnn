# image classification from scratch: https://keras.io/examples/vision/image_classification_from_scratch/
import os
import numpy as np
#import keras
import cv2
#from keras import layers
import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

'''
# filter out corrupt images
print("filtering corrupt images")
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages",folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path,fname)
        try:
            image = cv2.imread(fpath)
            cv2.imwrite(fpath,image)
        except Exception as e:
            print(f"deleting {fpath}: {e}")
            num_skipped += 1
            os.remove(fpath)
print(f"deleted {num_skipped} images")
'''

# generate training and validation datasets
image_size = (180,180)
batch_size = 64
train_ds,val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# apply augmentation to training dataset
data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# configuring for performance
train_ds = train_ds.map(lambda img,label: (data_augmentation(img),label),num_parallel_calls=tf.data.AUTOTUNE)

# rescale images
train_ds = train_ds.map(lambda img,label: (img / 255.0,label),num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda img,label: (img / 255.0,label),num_parallel_calls=tf.data.AUTOTUNE)

# prefetch
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# build model (use KerasTuner to optimize hyperparameters)
def make_model(input_shape,num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(128,3,strides=2,padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    previous_block_activation = x
    for size in [256,512,768]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size,3,padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size,3,padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(3,strides=2,padding="same")(x)
        residual = keras.layers.Conv2D(size,1,strides=2,padding="same")(previous_block_activation)
        x = keras.layers.add([x,residual])
        previous_block_activation = x
    x = keras.layers.SeparableConv2D(1024,3,padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(units,activation=None)(x)
    return keras.Model(inputs,outputs)

model = make_model(input_shape=image_size + (3,),num_classes=2)
#keras.utils.plot_model(model,show_shapes=True)

qa_model = tfmot.quantization.keras.quantize_model(model)

# train
epochs = 10

'''
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "cp{epoch}-{val_loss:.2f}.keras",
    ),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.summary()
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds
)
'''

qa_callbacks = [
    keras.callbacks.ModelCheckpoint(
        "qa_cp{epoch}-{val_loss:.2f}.keras",
    ),
]
qa_model.compile(optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
qa_model.summary()
qa_model.fit(
    train_ds,
    epochs=epochs,
    callbacks=qa_callbacks,
    validation_data=val_ds
)

# quantize
converter = tf.lite.TFLiteConverter.from_keras_model(qa_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# save
with open("model.tflite","wb") as file:
    file.write(quantized_model)
