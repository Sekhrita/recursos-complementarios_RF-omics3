# recursos-complementarios_RF-omics3

## Breve contexto

Este proyecto tiene como objetivo implementar diferentes versiones de un modelo de Random Forest para la clasificación de tumores de origen desconocido, utilizando datos de mutaciones de nucleótido único (SNP) obtenidos del GDC Data Portal. Se incluyen enfoques para mejorar el rendimiento del modelo, tales como el balanceo de datos y la optimización de hiperparámetros, con el fin de lograr una mejor precisión en la identificación del sitio primario del tumor.

## Objetivos de los códigos

- Entrenar y evaluar diferentes versiones del modelo Random Forest.

- Experimentar con técnicas de balanceo de datos y optimización de hiperparámetros para mejorar el rendimiento del modelo.

- Generar gráficos para evaluar el desempeño del modelo, como las curvas ROC y matrices de confusión.

## Intrucciones
En cuanto a las dependencias necesarias y al método de uso, a grandes rangos se utiliza lenguaje python (junto con librerías) para tratar los datos y entrenar el modelo.

### Cómo crear un entorno virtual e instalar los requirements.txt (OS Debian)

1.- Crear un entorno virtual:
```bash
python3 -m venv .env
```

2.- Activar el entorno virtual:
```bash
source .env/bin/activate
```

3.- Instalar las dependencias desde el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

### Cómo utilizar los códigos

1.- Activa el entorno virtual antes de ejecutar cualquier script, para asegurarte de que todas las dependencias estén disponibles.

2.- Ejecuta cada script utilizando Python desde la línea de comandos. Ejemplo:
```bash
python cross_val-random_forest.py
```

## Qué hace cada código

- `cross_val-random_forest.py`: Genera y entrena un modelo Random Forest con los hiperparámetros por defecto y sin balanceo en los casos.

- `cross_val-random_forest-optuna.py`: Genera y entrena un modelo Random Forest utilizando datos desbalanceados, pero optimizando los hiperparámetros mediante Optuna para mejorar el rendimiento.

- `cross_val-random_forest-SMOTE.py`: Genera y entrena un modelo Random Forest con datos balanceados mediante SMOTE (oversampling).

- `cross_val-random_forest-SMOTE-Optuna.py`: Genera y entrena un modelo Random Forest con datos balanceados mediante SMOTE y utilizando los mejores hiperparámetros encontrados con Optuna.

- `cross_val-random_forest-SMOTE-Optuna-grafs.py`: Similar al anterior, pero además genera gráficos de ROC y matrices de confusión para evaluar el desempeño del modelo.

- `CutData.py`: Este script se encuentra dentro de la carpeta `/data` y se utiliza para reducir el número de ejemplos de cada clase, balanceando las características mediante la eliminación de datos (undersampling). Si se desea utilizar un conjunto de datos reducido, basta con cambiar el archivo de datos que llama cada script por el nuevo archivo con datos recortados.

