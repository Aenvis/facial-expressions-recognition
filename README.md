

  <h3 align="center">Klasyfikacja wyrazów twarzy</h3>

  <p align="center">
	  Wojciech Czerwiński 147641 
	</p>
	  <p align="center">
   Deep Learning (Convolutional Neural Network)
   </p>
    <p align="center">
   Projekt na zaliczenie przedmiotu <b>Wprowadzenie do Sztucznej Inteligencji czwartek 11:45</b>
	</p>


## Spis treści

- [Opis problemu](#opis-problemu)
- [State of art](#state-of-art)
- [Opis wybranej koncepcji](#opis-wybranej-koncepcji)
- [Moja implementacja](#moja-implementacja)


## Opis problemu

Wybrany przeze mnie problem polega na klasyfikacji wyrazów twarzy wykorzystując <b>CNN - konwolucyjną sieć neuronową</b> - jest to rodzaj architektury algorytmów głębokiego uczenia oparty o warstwy splotowe, szczególnie wykorzystywany w przetwarzaniu danych graficznych.  
- ### Dane wejściowe 
		Zdjęcia twarzy w rozdzielczości 48x48, wszystkie zdjęcia są w skali szarości. 
		Zdjęcia przedstawiają twarze wyrażające 7 różnych emocji (klasy w modelu):
		angry, disgusted, fearful, happy, neutral, sad, surprised
- ### Zamierzony efekt
		Celem projektu jest klasyfikacja zdjęć podanych przez użytkownika na podstawie wytrenowanego modelu.
		Np. podając zdjęcie uśmiechniętej twarzy (może być dowolna rozdzielczość oraz paleta barw)
		model powinien opisać je klasą 'happy'
- ### Motywacja
		Głębokie uczenie z wykorzystaniem sieci neuronowych, a w szczególności CNN dla problemów 
		związanych z przetwarzaniem obrazów, jest przyszłościową technologią, która w znaczny
		sposób może poprawić jakość oraz przyspieszyć rozwój w wielu dziedzinach życia codziennego,
		np. przemysł, medycyna, lotnictwo.
## State of art
| **Wybrane typy uczenia maszynowego** | **opis działania** | Zalety | Wady |
|:---:|:---:|:---:|:---:|
| CNN - convolutional neural networks | Warstwy konwolucyjne ekstraktują cechy z obrazy,<br>a następnie przetwarzają je na warstwach<br>pełniących rolę klasyfikatora | - skuteczne w rozpoznawaniu obiektów i twarzy<br>- prostsza architektura niż generyczne DNN<br>- Warstwy konwolucyjne, które pozwalają na efektywną ekstrakcję cech | -Mogą być mniej skuteczne w rozpoznawaniu bardziej skomplikowanych obiektów<br>- duży koszt wykonywanych obliczeń |
| DNN - deep neural networks | Bardzo duża liczba warstw ukrytych, które<br>przetwarzają dane i w ten sposób są w stanie<br>trenować model | - skuteczne w rozpoznawaniu obiektów i twarzy a także generowaniu obrazów<br>- Architektura pozwala na uczelnie bardziej złożonych modeli | - Potrzebują bardzo dużą ilość danych<br>-Proces uczenia jest jeszcze dłuższy niż CNN |
| Uczenie przez ekstrakcję cech | Z danych wejściowych wyodrębniane<br> są kluczowe cechy,<br> które są następnie wykorzystane<br> do uczenia modelu  | - prostota, która pozwala na skupienie się na kluczowych cechach danych<br>- skuteczna, gdy cechy są dobrze opisane | - mniej skuteczna niż CNN i DNN |

## Opis wybranej koncepcji
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
## Moja implementacja
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

Potem model jest komplikowany i trenowany. Model w moim projekcie osiągnął bardzo złe wyniku, m.in. mocny overfitting. Mogło być to spowodowane zdjęciami wejściowymi słabej jakości, oraz nieodpowiednio dobranymi warstwami i parametrami sieci neuronowej. Aby minimalnie poprawić jakość modelu zrezygnowałem z klasy 'disgusted', która zawierała znacznie mniej zdjęć, niż inne klasy. Nieznacznie poprawiło to wskaźniki val i accuracy, których krzywe uczenia przedstawiłem poniżej (dla zbiorów danych treningowego i testowego):

![Figure_1](https://user-images.githubusercontent.com/76266906/213817023-a8522f20-70cd-4faf-968d-5c7350de49cf.png)
![Figure_2](https://user-images.githubusercontent.com/76266906/213817033-297a4ba1-d712-48b5-ba43-43cf0078da59.png)


