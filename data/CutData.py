import pandas as pd

# Cargar el archivo original
df = pd.read_csv('final_data_cleaned-full.tsv', sep='\t')  # Ajusta el delimitador si es necesario

# Identificar la columna de la clase objetivo
target_column = 'tissue_or_organ_of_origin'

# Definir el número de muestras deseado por clase
samples_per_class = 550

# Crear un DataFrame vacío para almacenar las muestras balanceadas
balanced_df = pd.DataFrame()

# Para cada clase, tomar una muestra aleatoria de "samples_per_class" elementos
for tissue_type in df[target_column].unique():
    tissue_df = df[df[target_column] == tissue_type]
    if len(tissue_df) >= samples_per_class:
        sampled_tissue_df = tissue_df.sample(n=samples_per_class, random_state=42)
    else:
        sampled_tissue_df = tissue_df  # Si hay menos de "samples_per_class", tomar todas las muestras disponibles
    balanced_df = pd.concat([balanced_df, sampled_tissue_df])

# Guardar el DataFrame balanceado en un nuevo archivo
balanced_df.to_csv('balanced_data.tsv', index=False, sep='\t')
