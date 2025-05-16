import pandas as pd
from data_processing import dataset_loading

def calculate_statistics(data = dataset_loading.load_dataset(), column_name=""):
    if column_name not in data.columns:
        return None, "Column not found"
    column_data = data[column_name]

    if column_data.empty:
        return None, "Only NaN/0."

    try:
        numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()

        if numeric_data.empty:
            return None, "Only NaN/0 in column"

        mean = numeric_data.mean()
        median = numeric_data.median()
        variance = numeric_data.var()
        std_dev = numeric_data.std()
        maximum = numeric_data.max
        minimum = numeric_data.min()

        stats = {
            'Математичне сподівання (Середнє)': mean,
            'Медіана': median,
            'Дисперсія': variance,
            'Середньоквадратичне відхилення': std_dev,
            'Максимальне значення': maximum,
            'Мінімальне значення': minimum
        }
        return stats, None
    except Exception as e:
        return None, f"[ERROR]: {e}"
