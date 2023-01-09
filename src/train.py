import tensorflow as tf
import pandas as pd

train_path = '../data/train'
test_path = '../data/test'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
# load train dataset
train_ds = train_datagen.flow_from_directory(
train_path,
target_size=(48, 48),
batch_size=64,
color_mode='grayscale',
class_mode='categorical'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
# load test dataset
test_ds = test_datagen.flow_from_directory(
test_path,
target_size=(48, 48),
batch_size=64,
color_mode='grayscale',
class_mode='categorical'
)

model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6), metrics=['accuracy'])
model_info = model.fit(
    train_ds,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=test_ds,
    validation_steps=7178 // 64
)

model.save_weights('../data/model.h5')
model.save_weights('../data/model1.h5')