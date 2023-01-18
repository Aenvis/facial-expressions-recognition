import os
import tensorflow as tf
import numpy as np
import cv2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
model = tf.keras.models.load_model('data/model1.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# It also preprocess loaded iamge
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.astype('float32') / 255.0
    img = img.reshape(1,48,48,1)
    return img

def load_images_from_dir(dir_path: str) -> str:
    imgs = []
    for file in os.listdir(dir_path):
        imgs.append(file)
    return imgs

def main():
    test_path = SRC_DIR+'/tests/my_face'
    imgs_paths = load_images_from_dir(test_path)
    imgs = [load_img(test_path+'/'+path) for path in imgs_paths]
    predictions = [model.predict(img) for img in imgs]
    class_idx = [np.argmax(prediction) for prediction in predictions]
    tf.keras.
    for i, val in enumerate(class_idx):
        print(f'{imgs_paths[i]} seems to be {classes[val]}!')

if __name__ == '__main__':
    main()