# spark-course-final
Este proyecto forma parte del curoso Big Data con Apache Spark de la Universidad de Ciencias Informaticas (UCI)
El mismo consiste en la asimilasion del las librerias de Apache Spark para el trabajo de Data Science.

1. Utilizando el fichero en formato csv provisto por su profesor realice las siguientes acciones:
a. Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10
primeras filas.
b. Imprima en pantalla los posibles valores que toma el atributo predictor o etiqueta
de clasificación.
c. Realice las transformaciones sobre los datos para eliminar valores ausentes,
datos anómalos, etc.
d. Aplique las transformaciones necesarias sobre los datos que contengan valores
nominales, mediante técnicas de extracción de características.

2. Seleccione al menos tres algoritmos de aprendizaje automático de acuerdo al problema
identificado en el dataset y realice las siguientes acciones:
a. Cree un VectorAssembler a partir de los datos pre-procesados y divida de forma
aleatoria el conjunto en dos partes un 70 % para entrenamiento y el 30 % para
pruebas.
b. Entrene cada modelo elegido con dicha entrada y ajuste los hiper-parámetros
correspondientes de forma automática.
c. Evalúe el resultado del entrenamiento de cada algoritmo mediante el conjunto de
pruebas. Muestre su accuracy y matriz de confusión.
d. Salve en un fichero el modelo que mejor resultados arrojó. 