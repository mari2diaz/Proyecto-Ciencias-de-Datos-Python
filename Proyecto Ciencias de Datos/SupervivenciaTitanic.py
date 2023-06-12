import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Importar los datos
dt_train = pd.read_csv('train.csv')  # Leer el archivo train.csv y guardarlo en el DataFrame dt_train
dt_test = pd.read_csv('test.csv')  # Leer el archivo test.csv y guardarlo en el DataFrame dt_test

# Preprocesamiento de los datos
x_train = dt_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy()  # Seleccionar las columnas relevantes del DataFrame dt_train y copiarlas a x_train
y_train = dt_train['Survived']  # La columna 'Survived' del DataFrame dt_train es la variable objetivo, se guarda en y_train
x_test = dt_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy()  # Seleccionar las columnas relevantes del DataFrame dt_test y copiarlas a x_test

# Reemplazar datos faltantes
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)  # Reemplazar los valores faltantes en la columna 'Age' de x_train con el valor promedio de la columna
x_test['Age'].fillna(x_train['Age'].mean(), inplace=True)  # Reemplazar los valores faltantes en la columna 'Age' de x_test con el valor promedio de la columna de x_train

# Convertir variables categóricas a numéricas
x_train.loc[:, 'Sex'].replace({'female': 0, 'male': 1}, inplace=True)  # Reemplazar los valores 'female' por 0 y 'male' por 1 en la columna 'Sex' de x_train
x_test.loc[:, 'Sex'].replace({'female': 0, 'male': 1}, inplace=True)  # Reemplazar los valores 'female' por 0 y 'male' por 1 en la columna 'Sex' de x_test

# Crear instancia de KNN
knn = KNeighborsClassifier(n_neighbors=3)  # Crear un clasificador KNN con 3 vecinos

# Entrenar el modelo
knn.fit(x_train, y_train)  # Entrenar el modelo KNN utilizando x_train como características y y_train como variable objetivo

# Realizar predicciones
predictions = knn.predict(x_test)  # Utilizar el modelo entrenado para hacer predicciones sobre x_test

# Mostrar los resultados por pantalla
output = pd.DataFrame({'PassengerId': dt_test['PassengerId'], 'Survived': predictions})  # Crear un DataFrame de salida con las columnas 'PassengerId' y 'Survived'
print(output.head())  # Mostrar las primeras filas del DataFrame de salida