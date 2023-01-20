import tensorflow as tf
import pandas as pd

train_path = 'data/train'
test_path = 'data/test'

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
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=8, min_lr=0.00001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
model_info = model.fit(
    train_ds,
    epochs=50,
    callbacks=[reduce_lr, early_stop],
    validation_data=test_ds,
)

print(model_info.history)
model.save('data/model1.h5') 