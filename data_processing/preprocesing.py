import numpy as np
import pandas as pd
from data_processing import dataset_loading
from sklearn.preprocessing import StandardScaler # Для нормалізації


def process_data():
    data = dataset_loading.load_dataset()

    data_processed = data.fillna(0)

    return data_processed


def normalize_column(df, column_name):
    if column_name not in df.columns:
        print(f"Помилка нормалізації стовпця: Стовпець '{column_name}' не знайдено.")
        return df.copy()

    df_normalized = df.copy()

    col_to_normalize = pd.to_numeric(df_normalized[column_name], errors='coerce')

    valid_indices = col_to_normalize.dropna().index
    data_for_scaling = col_to_normalize.loc[valid_indices].values.reshape(-1, 1)

    if data_for_scaling.size == 0:
        print(f"Помилка нормалізації стовпця: Стовпець '{column_name}' не містить числових даних для нормалізації.")
        return df.copy()

    try:
        scaler = StandardScaler()

        normalized_data = scaler.fit_transform(data_for_scaling)

        # Створюємо нормалізовану серію з оригінальними індексами
        normalized_series = pd.Series(normalized_data.flatten(), index=valid_indices, name=f'{column_name}_Normalized_')

        old_col_name_standard = f'{column_name}_Normalized_standard'
        df_normalized = df_normalized.drop(columns=[old_col_name_standard])


        df_normalized = df_normalized.join(normalized_series)

        return df_normalized

    except Exception as e:
        print(f"Помилка нормалізації стовпця для стовпця '{column_name}': {e}")
        return df.copy()


def normalize_country_population(population_series):

    data_for_scaling = np.array(population_series).reshape(-1, 1)


    nan_indices = np.isnan(data_for_scaling).flatten()
    valid_data = data_for_scaling[~nan_indices]

    if valid_data.size == 0:
        print("Помилка нормалізації рядка: Немає числових даних для нормалізації.")

        return np.array(population_series), (None, None)

    try:
        scaler = StandardScaler()

        normalized_valid_data = scaler.fit_transform(valid_data)

        normalized_full_data = np.full(data_for_scaling.shape, np.nan)
        normalized_full_data[~nan_indices] = normalized_valid_data

        return normalized_full_data.flatten(), scaler

    except Exception as e:
        print(f"[ERROR]: {e}")
        return np.array(population_series), (None, None)