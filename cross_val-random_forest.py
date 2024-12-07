import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Cargar el archivo
input_file = "data/final_data_cleaned-full.tsv"
data = pd.read_csv(input_file, sep="\t", encoding="utf-8-sig")

# Convertir 'gender' a valores numéricos
data["gender"] = data["gender"].map({"male": 1, "female": 2})

# Verificar si hay valores nulos en 'gender'
if data["gender"].isnull().any():
    print("Advertencia: Hay valores nulos en la columna 'gender'")
    data = data.dropna(subset=["gender"])

# Variables predictoras y objetivo
X = data.drop(columns=["case_id", "Patient_ID", "tissue_or_organ_of_origin"])
y = data["tissue_or_organ_of_origin"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Implementar validación cruzada
print("Evaluando modelo con validación cruzada...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

# Mostrar los resultados de la validación cruzada
print("Resultados de la validación cruzada:")
print("Scores por cada pliegue:", cv_scores)
print("Media de accuracy:", np.mean(cv_scores))
print("Desviación estándar de accuracy:", np.std(cv_scores))

# Entrenar el modelo con todos los datos de entrenamiento
print("Entrenando el modelo con todos los datos de entrenamiento...")
rf_model.fit(X_train, y_train)

# Hacer predicciones
print("\nEvaluando modelo...")
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
print("\nResultados del Modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(rf_model, "random_forest_model.joblib")
print("Modelo guardado como 'random_forest_model.joblib'")

# Analizar las características más importantes con SHAP
print("\nAnalizando las características más importantes con SHAP...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# Crear un resumen de las características importantes y guardarlo como imagen
shap.summary_plot(shap_values, X_train, show=False, plot_size=(12, 6))
plt.title("SHAP Summary Plot for all Cancer Type", fontsize=12, pad=30)
plt.savefig("shap_summary_plot_all-cancer.png")
plt.close()

# Crear gráficos de resumen de SHAP para diferentes tipos de cáncer
cancer_types = y_train.unique()
for cancer in cancer_types:
    print(f"\nCreando gráfico SHAP para el cáncer de tipo: {cancer}")
    indices = y_train[y_train == cancer].index
    X_cancer = X_train.loc[indices]
    shap_values_cancer = explainer.shap_values(X_cancer)
    
    # Crear y guardar el gráfico de resumen para el tipo de cáncer actual
    shap.summary_plot(shap_values_cancer, X_cancer, show=False, plot_size=(12, 6))
    plt.title(f"SHAP Summary Plot for Cancer Type: {cancer}", fontsize=12, pad=30)
    plt.savefig(f"shap_summary_plot_{cancer}.png")
    plt.close()

# Crear un entorno virtual (env) con las librerías necesarias
# Ejecuta los siguientes comandos en la terminal:
# python -m venv env
# source env/bin/activate (en Linux/macOS) o .\env\Scripts\activate (en Windows)
# pip install pandas scikit-learn tqdm numpy joblib shap matplotlib