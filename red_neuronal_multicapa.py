# -*- coding: utf-8 -*-
"""
MLP classifier with 4 hidden layers
Author: Juan Elier Rosales Rosas
Universidad Autonoma Metropolitana unidad Lerma
Matricula: 2172040721
"""
#Importamos las bibliotecas que vamos a usar
#Pandas para el manejo y analisis de estructuras de datos
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
#Lee el archivo csv y lo asigna en la variable file
file=pd.read_csv('C:/Users/elier/Documents/Trimestre 21-P/Introducción a la inteligencia artificial/Tareas/heart.csv')
#imprime la informacion del archivo como tipos de datos que ocupan las variables y las columnas que utiliza
print(file.info())
print("\n"*2)
#Organizamos con pandas convirtiendo en DataFrame.
heart=pd.DataFrame(file)
#Imprime las estadisticas del archivo.
print("Estadisticas")
print(heart.describe())
print("\n"*2)
#Vamos a calcular las estadisticas del archivo variable por variable.
print("Estadisticas seccion por seccion")
print("\n"*2)
print("Edad")
print(heart['age'].describe())
print("\n"*2)
print("Sexo")
print(heart['sex'].describe())
print("\n"*2)
print("cp")
print(heart['cp'].describe())
print("\n"*2)
print("trestbps")
print(heart['trestbps'].describe())
print("\n"*2)
print("chol")
print(heart['chol'].describe())
print("\n"*2)
print("fbs")
print(heart['fbs'].describe())
print("\n"*2)
print("restcg")
print(heart['restecg'].describe())
print("\n"*2)
print("thalach")
print(heart['thalach'].describe())
print("\n"*2)
print("exang")
print(heart['exang'].describe())
print("\n"*2)
print("oldpeak")
print(heart['oldpeak'].describe())
print("\n"*2)
print("ca")
print(heart['ca'].describe())
print("\n"*2)
print("slope")
print(heart['slope'].describe())
print("\n"*2)
print("thal")
print(heart['thal'].describe())
print("\n"*2)
print("target")
print(heart['target'].describe())

#Preprocesamiento
feature_names = ['sex','cp']
X = heart[feature_names]
y = heart['target']

#Divide datos de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=.4)
#Utilizamos el escalador MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''Preparamos el modelo Multi layer perceptron con 4 capas ocultas
con 100,40,60,50 neuronas respectivamente, tenemos un numero maximo de 1000 iteraciones
y una funcion de activacion rectified linear unit y ademas entrenamos el modelo
'''
clf = MLPClassifier(random_state=0,hidden_layer_sizes=(100,40,60,50,),max_iter=1000, activation='relu').fit(X_train, y_train)
#Imprime los primeros 10 datos de la salida deseada
print("Salida deseada", y_test[:10])
print("Salida obtenida", clf.predict(X_train))
print("Precision", clf.score(X_test,y_test))