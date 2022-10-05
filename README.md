# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Полторапавлова Александра Алексеевича
- 2093937
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

Ссылка на код: https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Google_colab_scripts/linear_regression.ipynb

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
![image](https://user-images.githubusercontent.com/98959447/194002877-13868740-9794-4fae-9a9d-60cde41638bd.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task1(1).png

- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
```py
# Функция модели: определяет модель линейной регрессии wx+b
def model(w, b, x):
    return w*x + b

# Функция потерь: функция потерь среднеквадратичной ошибки
def loss_function(w, b, x, y):
    num = len(x)
    prediction = model(w, b, x)
    return (0.5/num) * (np.square(prediction - y)).sum()

# Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
def optimize(w, b, x, y):
    num = len(x)
    prediction = model(w, b, x)
    dw = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * ((prediction - y).sum())
    
    w = w - Lr*dw
    b = b - Lr*db
    return w, b

def iterate(w, b, x, y, times):
    for i in range (times):
        w, b = optimize (w, b, x, y)
        
    return w, b
```
![image](https://user-images.githubusercontent.com/98959447/194003345-1a5dc310-5fb0-4224-a997-c3895919b5bb.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task1(2).png

```py
w = np.random.rand(1)
print(f'w = {w}') # написал, что w есть w, а b есть b, чтобы не путаться
b = np.random.rand(1)
print(f'b = {b}')
Lr = 0.000001

w, b = iterate(w, b, x, y, 1000000) # the more times it iterates the line is more correct
prediction = model(w, b, x)
loss = loss_function(w, b, x, y)
print({'w': w, 'b': b, 'loss': loss}) # made dict for my understanding
plt.scatter(x, y)
plt.plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/98959447/194004381-9bb72289-ec1c-424a-9642-a2b0fff460fd.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task1(3).png


## Задание 2
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

При определенных начальных данных величина loss стремится к нулю. loss -> 0 при условии, что сумма элементов массива 'y' равняется сумме элементов массива 'x', умноженных на 'w' и еще прибавить 'b'. При этом длина массивов не имеет значение

Ссылка на код: https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Google_colab_scripts/linear_regression.ipynb

```py
def is_loss_zero():
#  (0.5/10) * np.square(w*x + b - y) -> 0
#  np.square(w*x + b - y) -> 0
# w*x + b - y -> 0
# w*x -> y - b
# if w*x -> y - b then loss -> 0
  x = 100
  y = w * x + b
  return (0.5/10) * np.square(w*x + b - y)

print (is_loss_zero()) # loss -> 0 при условии, что сумма элементов 
# массива 'y' равняется сумме элементов массива 'x', умноженных на 'w' и 
# еще прибавить 'b'. При этом длина массивов не имеет значение

```
![image](https://user-images.githubusercontent.com/98959447/194005285-48a553c1-c425-4a34-803f-17bf27f346fe.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task2.png

### Задание 3
## Написать "Hello world" На python и в Unity 3D

Ссылка на код: https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Google_colab_scripts/Hello_world.ipynb

Код на python:
```py
print('Hello world!')
```

![image](https://user-images.githubusercontent.com/98959447/194006464-a9e22eca-b231-4cf5-bf59-46904c28ddd0.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task3(1).png

Ссылка на код: https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Work_with_unity/New%20Unity%20Project/Assets/Scripts/Hello_world.cs

Код на Unity:
```cs
using UnityEngine;

public class Hello_world : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello world!");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

```

![image](https://user-images.githubusercontent.com/98959447/194013395-5fd4dbdc-fe63-455e-b60d-16e605594f85.png)
https://github.com/AlexPoltorapavlov/ML_Labs_and_Homeworks/blob/main/Lab1/Photo_report/Task3(2)png.png

## Выводы
Ознакомился с принципом работы линейной регрессии, вспомнил немного математики для второго задания. Ознакомился немного с Unity и языком C#.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
