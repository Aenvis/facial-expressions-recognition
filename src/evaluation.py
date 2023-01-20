import tensorflow as tf
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

train_path = 'data/train'
test_path = 'data/test'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# load train dataset
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=(48, 48),
batch_size=64,
color_mode='grayscale',
class_mode='categorical'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# load test dataset
test_generator = test_datagen.flow_from_directory(
test_path,
target_size=(48, 48),
batch_size=64,
color_mode='grayscale',
class_mode='categorical'
)

model = tf.keras.models.load_model('data/model1.h5')

class Set(Enum):
    TRAIN = 0,
    TEST = 1

# loss and accuracy
train_loss, train_acc = model.evaluate(train_generator, batch_size=64)
test_loss, test_acc = model.evaluate(test_generator, batch_size=64)

def GetLoss(set: Set) -> float:
    return train_loss if set is Set.TRAIN else test_loss

def GetAccuracy(set: Set) -> float:
    return train_acc if set is Set.TRAIN else test_acc

# confusion matrix
train_predictions = model.predict(train_generator)
train_label = train_generator.classes
train_label_pred = [np.argmax(pred) for pred in train_predictions]

cm_train = tf.math.confusion_matrix(train_label, train_label_pred)

test_predictions = model.predict(test_generator)
test_label = test_generator.classes
test_label_pred = [np.argmax(pred) for pred in test_predictions]

cm_test = tf.math.confusion_matrix(test_label, test_label_pred)

def PlotConfusionMatrix(set: Set) -> None:
    plt.imshow(cm_test if set is Set.TEST else cm_train, cmap='binary')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()