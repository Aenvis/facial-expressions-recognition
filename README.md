


  <h3 align="center">Facial expressions recognition</h3>

  <p align="center">
	  Wojciech Czerwiński 147641
	</p>
	  <p align="center">
   Deep Learning (Convolutional Neural Network)
   </p>
    <p align="center">
   <b>course: Intro to Artificial Intelligence @ Poznań University of Technology</b>
	</p>


## Table of contents

- [About the project](#about-the-problem)
- [State of art](#state-of-art)
- [Description of chosen solution](#description-of-chosen-solution)
- [Implementation](#implementation)


## About the project

The chosen problem is to classify facial expressions using a <b>CNN - Convolutional Neural Network</b> - a type of deep learning algorithm architecture based on convolutional layers, particularly used in image data processing. 
- ### Input 
		Images of faces in 48x48 resolution, all images are in greyscale. 
		Images show faces expressing 7 different emotions (classes in the model):
		angry, disgusted, fearful, happy, neutral, sad, surprised
- ### Desired effect
		The purpose of the project is to classify the images given by the user based on a trained model.
		E.g. giving an image of a smiling face (can be any resolution and color palette)
		the model should describe it with the class 'happy'
- ### Motivation 
		Deep learning using neural networks, and in particular CNNs for problems 
		related to image processing, is a future-oriented technology that can significantly 
		improve quality and accelerate development in many areas of everyday life,
		e.g. industry, medicine, aviation.
## State of art
| **Wybrane typy uczenia maszynowego** | **opis działania** | Zalety | Wady |
|:---:|:---:|:---:|:---:|
| CNN - convolutional neural networks | Warstwy konwolucyjne ekstraktują cechy z obrazy,<br>a następnie przetwarzają je na warstwach<br>pełniących rolę klasyfikatora | - skuteczne w rozpoznawaniu obiektów i twarzy<br>- prostsza architektura niż generyczne DNN<br>- Warstwy konwolucyjne, które pozwalają na efektywną ekstrakcję cech | -Mogą być mniej skuteczne w rozpoznawaniu bardziej skomplikowanych obiektów<br>- duży koszt wykonywanych obliczeń |
| DNN - deep neural networks | Bardzo duża liczba warstw ukrytych, które<br>przetwarzają dane i w ten sposób są w stanie<br>trenować model | - skuteczne w rozpoznawaniu obiektów i twarzy a także generowaniu obrazów<br>- Architektura pozwala na uczelnie bardziej złożonych modeli | - Potrzebują bardzo dużą ilość danych<br>-Proces uczenia jest jeszcze dłuższy niż CNN |
| Uczenie przez ekstrakcję cech | Z danych wejściowych wyodrębniane<br> są kluczowe cechy,<br> które są następnie wykorzystane<br> do uczenia modelu  | - prostota, która pozwala na skupienie się na kluczowych cechach danych<br>- skuteczna, gdy cechy są dobrze opisane | - mniej skuteczna niż CNN i DNN |

## Description of chosen solution
Jak przedstawiono w poprzednich punktach, <b> konwolucyjne sieci neuronowe </b> to rodzaj sieci neuronowych, które są specjalnie zaprojektowane do przetwarzania obrazów. Działanie CNN polega na przeprowadzeniu konwolucji (czyli operacji splotu) danych wejściowych z filtrem, co pozwala na wykrycie wzorców w obrazie.

Architektura CNN składa się z kilku warstw:
1.  Warstwa konwolucyjna: ta warstwa składa się z filtrów, które służą do wykrywania wzorców w obrazie. Filtry przesuwają się po obrazie i przeprowadzają konwolucję z danymi wejściowymi, co pozwala na wykrycie wzorców w różnych częściach obrazu.
2.  Warstwa max-pooling: ta warstwa polega na redukcji rozmiaru danych wejściowych poprzez zastosowanie operacji max-pooling (zamiana kilku pikseli na jeden, zachowując tylko największą wartość).
3.  Warstwy ukryte: po warstwie max-pooling następują warstwy ukryte, które przetwarzają cechy z warstw konwolucyjnych i max-pooling.
4.  Warstwa wyjściowa: ta warstwa jest odpowiedzialna za klasyfikację obrazu na podstawie cech wyodrębnionych przez warstwy wcześniejsze.

Jako <b>dane wejściowe</b> potrzebne są zdjęcia twarzy. Zbiory danych w postaci zdjęć w celu trenowania modeli sztucznej inteligencji dostępne są w internecie. Do mojego problemu wykorzystałem:
- https://www.kaggle.com/datasets/msambare/fer2013

<b> Wyjściem algorytmu </b> dla każdego zdjęcia jest lista tylu wartości w przedziale < 0.0 ; 1.0 >, ilu klas danych wejściowych. Każda kolejna wartość odpowiada predykcji na ile algorytm ocenia, że zdjęcie należy do klasy o tym indeksie. Przykład:
Jeśli algorytm zwróci [0.013807331, 0.003256767, 0.9232555, 0.02688416, 0.028897915, 0.0038983417], oznacza to, że w następujący sposób klasyfikuje zdjęcie na wejściu:

<b>0.013807331 * 100%</b>, że jest to klasa <b>angry</b>

<b>0.003256767* 100%</b>, że jest to klasa <b>fearful</b>

<b>0.9232555* 100%</b>, że jest to klasa <b>happy</b>

<b>0.02688416* 100%</b>, że jest to klasa <b>neutral</b>

<b>0.028897915* 100%</b>, że jest to klasa <b>sad</b>

<b>0.0038983417* 100%</b>, że jest to klasa <b>surprised</b> 

Pominięto jedną klasę ze względu na implementację.

Aby zastosować architekturę CNN, potrzeba bardzo dużych zasobów obliczeniowych oraz dużej odpowiednio przygotowanych ilości danych, aby model był skuteczny oraz jak jak najbardziej niezawodny. Potrzebny jest także dobrze zaprojektowany model, co jest bardzo trudnym zadaniem i zależy mocno od rozwiązywanego problemu.
## Implementation
W mojej implementacji korzystałem z biblioteki TensorFlow.

### Etapy projektu
- #### Uczenie modelu
	 Załadowano zbiory danych treningowy oraz testowy. Dane <b> nie zostały </b> poddane preprocessingowi, co mogło wpłynąć na bardzo słabe wyniku modelu. Zrezygnowałem z preprocessingu, ze względu na to, że dane były już wcześniej przygotowane pod proces uczenia. Następnie, w oparciu o dostępne materiały w internecie, zaimplementowałem następującą sieć neuronową:
```python
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
```

Potem model jest komplikowany i trenowany. Model w moim projekcie osiągnął bardzo złe wyniku, m.in. mocny overfitting. Mogło być to spowodowane zdjęciami wejściowymi słabej jakości, oraz nieodpowiednio dobranymi warstwami i parametrami sieci neuronowej. Aby minimalnie poprawić jakość modelu zrezygnowałem z klasy 'disgusted', która zawierała znacznie mniej zdjęć, niż inne klasy. Nieznacznie poprawiło to wskaźniki loss i accuracy, których krzywe uczenia przedstawiłem poniżej (dla zbiorów danych treningowego i testowego):

![Figure_1](https://user-images.githubusercontent.com/76266906/213817023-a8522f20-70cd-4faf-968d-5c7350de49cf.png)
![Figure_2](https://user-images.githubusercontent.com/76266906/213817033-297a4ba1-d712-48b5-ba43-43cf0078da59.png)

Od pewnego wczesnego momentu krzywa dla danych testowych utrzymuje się na stałym poziomie.

Wskaźniki liczbowe loss i accuracy tak wytrenowanego modelu:

```Loss for the train data set: 0.2262008637189865
Loss for the test data set: 1.133596420288086
Accuracy for the train data set: 0.96692955493927
Accuracy for the test data set: 0.6197820901870728
```
Jak można odczytać z powyższych wskaźników, wskaźnik dokładności dla danych treningowych wynosi ~0.97, gdzie dla danych testowych ~0.62.

Model poddałem także ewaluacji poprzez macierz pomyłek, które wyglądają następująco:

- zbiór danych treningowych
![macierz1](https://user-images.githubusercontent.com/76266906/213818756-9fe8beab-fbb5-4b70-acc1-7446ec4629b8.png)

- zbiór danych testowych
![macierz2](https://user-images.githubusercontent.com/76266906/213818765-e45614b7-4516-41f1-a7b8-e04671fc1860.png)


Widać zdecydowaną dominację klasy o indeksie 2 - fearful, zarówno w danych treningowych jak i testowych.
