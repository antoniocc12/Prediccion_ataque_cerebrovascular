pres='''
Es una hemorragia dentro de un órgano o una pérdida de la 
circulación sanguínea que se dirige hacia un órgano, por un coágulo de sangre que 
tapona un vaso sanguíneo. Se suele se referir a una hemorragia cerebral.
'''
muertes='''
Según la Organización Mundial de la Salud, el accidente cerebrovascular es la 
segunda causa principal de muerte a nivel mundial, responsable de aproximadamente 
el 11% del total de muertes.
'''
link='''
Competición de Kaggle: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
'''
atributos='''
1) id: identificador único\n
2) género: "Masculino", "Femenino" u "Otro"\n
3) edad: edad del paciente\n
4) hipertensión: 0 si el paciente no tiene hipertensión, 1 si el paciente tiene hipertensión\n
5) heart_disease: 0 si el paciente no tiene ninguna enfermedad cardíaca, 1 si el paciente tiene una enfermedad cardíaca\n
6) alguna vez casado: "No" o "Sí"\n
7) work_type: "niños", "Govt_jov", "Nunca_trabajado", "Privado" o "Trabajador por cuenta propia"\n
8) Residence_type: "Rural" o "Urban"\n
9) avg_glucose_level: nivel medio de glucosa en sangre\n
10) bmi: índice de masa corporal\n
11) smoking_status: "anteriormente fumado", "nunca fumado", "fuma" o "Desconocido"\n
12) accidente cerebrovascular: 1 si el paciente tuvo un accidente cerebrovascular o 0 si no\n
'''
arreglar='''
Varias formas de arreglar los valores nulos:
- Se pueden eliminar estos registros
- Llenar los espacios en blanco con la media, la mediana o con el registro de antes o después del valor faltante.
- Usar un árbol de decisiones para predecir los faltantes
- Usar el método K-Neighbors para llenar los vacíos

En este caso, creamos una función para detectar los valores nulos y los sustituimos por el valor medio.
''' 
desc='''
- La media de las edades es alta (43,2 años), con una desviación típica de 22,61.
- Los valores medios de glucosa en sangre presentan una desviación mayor, puesto que son valores más elevados.
- La media del índice de masa corporal es de 28,89, que podemos considerar alto acorde con los datos de las edades. 
'''
report='''
- Alto valor del recall a la hora de predecir 0, y bajo con 1. 
- Indica que a pesar de tener un Accuracy alto, primamos la precisión al recall, puesto que nos dejamos muchos casos de ataques sin predecir. 
'''
mat='''
Reflejo de lo anterior, donde la gran mayoría de los negativos los predice bien, aunque lo contrario pasa con los positivos.
'''
roc='''
- Representación gráfica de la sensibilidad frente a la especificidad para un sistema clasificador binario.
- Posición adecuada acercándose a la esquina superior izquierda, pero sin caer en overfitting.
- Valor de 0.767.
'''