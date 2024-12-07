import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import optuna

# Cargar el archivo
input_file = "data/balanced_data-550.tsv"
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar SMOTE al conjunto de entrenamiento para balancear las clases
print("Aplicando SMOTE para balancear las clases en el conjunto de entrenamiento...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Definir la función objetivo para la optimización con Optuna
def objective(trial):
    # Definir los hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Crear el modelo Random Forest con los parámetros sugeridos
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Evaluar el modelo usando validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# Crear el estudio de Optuna y optimizar los hiperparámetros
study = optuna.create_study(direction='maximize')

# Optimizar el estudio y registrar cada prueba
study.optimize(objective, n_trials=100, catch=(Exception,))

# Mostrar los mejores parámetros encontrados
print("\nMejores hiperparámetros:")
print(study.best_params)

# Crear el modelo Random Forest con los mejores hiperparámetros
tuned_rf_model = RandomForestClassifier(
    n_estimators=study.best_params['n_estimators'],
    max_depth=study.best_params['max_depth'],
    min_samples_split=study.best_params['min_samples_split'],
    min_samples_leaf=study.best_params['min_samples_leaf'],
    random_state=42
)

# Implementar validación cruzada con los mejores hiperparámetros
print("Evaluando modelo con validación cruzada...")
cv_scores_best_params = cross_val_score(tuned_rf_model, X_train, y_train, cv=5, scoring='accuracy')

# Mostrar los resultados de la validación cruzada
print("Resultados de la validación cruzada:")
print("Scores por cada pliegue:", cv_scores_best_params)
print("Media de accuracy:", np.mean(cv_scores_best_params))
print("Desviación estándar de accuracy:", np.std(cv_scores_best_params))

# Entrenar el modelo con todos los datos de entrenamiento
print("\nEntrenando el modelo con todos los datos de entrenamiento...")
tuned_rf_model.fit(X_train, y_train)

# Hacer predicciones
print("\nEvaluando modelo...")
y_pred = tuned_rf_model.predict(X_test)

# Evaluar el modelo
print("\nResultados del Modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(tuned_rf_model, "tuned_random_forest_model.joblib")
print("Modelo guardado como 'tuned_random_forest_model.joblib'")

# Analizar las características más importantes con SHAP
print("\nAnalizando las características más importantes con SHAP...")
explainer = shap.TreeExplainer(tuned_rf_model)
shap_values = explainer.shap_values(X_train)

# Crear un resumen de las características importantes y guardarlo como imagen
print("\nCreando gráfico SHAP para el cáncer de tipo: todos")
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
