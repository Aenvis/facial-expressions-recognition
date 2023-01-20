import os
import tensorflow as tf
import numpy as np
import cv2
from evaluation import GetAccuracy, GetLoss, PlotConfusionMatrix, Set

# source directory absolute path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------
# Directory of images to be tested
# ---------------------------------------------
test_path = SRC_DIR+'/../data/test/happy'


# It also preprocess loaded iamge
def load_img(path) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.astype('float32') / 255.0
    img = img.reshape(1,48,48,1)
    return img

def load_images_from_dir(dir_path: str) -> float:
    imgs_paths = []
    for file in os.listdir(dir_path):
        imgs_paths.append(file)
    return [load_img(test_path+'/'+path) for path in imgs_paths]

def main():
    classes = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    model = tf.keras.models.load_model('data/model1.h5')

    imgs = load_images_from_dir(test_path)
    predictions = [model.predict(img) for img in imgs]
    class_idx = [np.argmax(prediction) for prediction in predictions]
    
    '''
    Scores for specific facial expressions. E.g. if we choose angry faces as input, it will tell you how well the model predicted.
    model1.h5 almost 100% for the train data set and not even 50 (49%) for the test data set
    '''
    #data = [id for id in class_idx if  id == 0]
    #print(len(data)/len(class_idx))
    
    # Print prediction for each input image - it's not always working
    #for i, val in enumerate(class_idx):
    #    print(f'Face {i} seems to be {classes[val]}!')

    # evaluation
    print('Loss for the train data set: ' + str(GetLoss(Set.TRAIN)))
    print('Loss for the test data set: ' + str(GetLoss(Set.TEST)))
    print('Accuracy for the train data set: ' + str(GetAccuracy(Set.TRAIN)))
    print('Accuracy for the test data set: ' + str(GetAccuracy(Set.TEST)))

    PlotConfusionMatrix(Set.TRAIN)

if __name__ == '__main__':
    main()