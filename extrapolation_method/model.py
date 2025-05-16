import numpy as np
import pandas as pd

from data_processing import preprocesing

class PolynomialExtrapolationModel:
    def __init__(self, data=preprocesing.process_data(), x_future=2025, degree=2, test_year=2022, test_country="Ukraine"):
        self.df = data
        self.x_future = x_future
        self.degree = degree
        self.test_year = test_year
        self.test_country = test_country
        self.results = []

    def model(self, calculate_error=False):
        years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
        results_list = []

        for index, row in self.df.iterrows():
            population_data = [row[f'{year} Population'] for year in years]

            valid_years = [years[i] for i, pop in enumerate(population_data) if pop > 0]
            valid_population = [pop for pop in population_data if pop > 0]

            extrapolated_population = np.nan
            coefficients = None

            if len(valid_years) >= self.degree + 1:
                try:
                    coefficients = np.polyfit(valid_years, valid_population, self.degree)
                    extrapolated_population = np.polyval(coefficients, self.x_future)
                except Exception as e:
                    print(f"Помилка polyfit для {row['Country/Territory']}: {e}")
                    extrapolated_population = np.nan
            else:
                extrapolated_population = np.nan

            result = {
                'Country/Territory': row['Country/Territory'],
                'Extrapolated Population': extrapolated_population
            }

            if calculate_error and f'{self.x_future} Population' in row:
                actual = row[f'{self.x_future} Population']
                if not pd.isna(actual) and actual != 0:
                    result['Actual Population'] = actual
                    result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
                    result['Percentage Error'] = (abs(extrapolated_population - actual) / actual * 100) if not pd.isna(extrapolated_population) and not pd.isna(actual) else np.nan
                else:
                    result['Actual Population'] = actual
                    result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
                    result['Percentage Error'] = np.inf if actual == 0 and not pd.isna(extrapolated_population) else np.nan

            results_list.append(result)

        self.results = results_list
        return pd.DataFrame(self.results)