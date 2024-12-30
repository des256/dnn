# image classification from scratch: https://keras.io/examples/vision/image_classification_from_scratch/
import os
import numpy as np
#import keras
import cv2
#from keras import layers
import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import tf2onnx
import onnx

AUGMENTATION_RANDOM = 0.2
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 3e-4
OUTPUT_PATH = "catdog.tflite"
ONNX_PATH = "../catdog.onnx"

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
train_ds,val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    interpolation="bicubic",
    crop_to_aspect_ratio=True,
)

# apply augmentation to training dataset
data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(AUGMENTATION_RANDOM),
    keras.layers.RandomZoom(AUGMENTATION_RANDOM),
    keras.layers.RandomContrast(AUGMENTATION_RANDOM),
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

# build model (TODO: use KerasTuner to optimize hyperparameters)
input_shape = image_size + (3,)
def make_model():
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
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs,outputs)

model = make_model()
#keras.utils.plot_model(model,show_shapes=True)

# train
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "cp{epoch}-{val_loss:.2f}.keras",
    ),
]
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.summary()
model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds
)

'''
qa_model = tfmot.quantization.keras.quantize_model(model)

qa_callbacks = [
    keras.callbacks.ModelCheckpoint(
        "qa_cp{epoch}-{val_acc:.2f}.keras",
    ),
]
qa_model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
qa_model.summary()
qa_model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=qa_callbacks,
    validation_data=val_ds
)

# quantize
converter = tf.lite.TFLiteConverter.from_keras_model(qa_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# save as TFLite
with open(OUTPUT_PATH,"wb") as file:
    file.write(quantized_model)
'''

# convert to ONNX
tf2onnx.convert.from_keras(model,output_path=ONNX_PATH,opset=16)

# manually remove ai.onnx.ml domain so wonnx can use it...
onnx_model = onnx.load(ONNX_PATH)
if len(onnx_model.opset_import) == 2:
    print("opset imports before: ",onnx_model.opset_import)
    onnx_model.opset_import.pop()
    print("opset imports after: ",onnx_model.opset_import)
onnx.save(onnx_model,ONNX_PATH)
