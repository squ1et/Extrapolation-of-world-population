import numpy as np
import pandas as pd


def execute_country_forecast(model):
    country_row = model.df[model.df['Country/Territory'] == model.test_country]

    if country_row.empty:
        return None, None

    country_row = country_row.iloc[0]
    years_known = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
    population_data = [country_row[f'{year} Population'] for year in years_known]

    valid_years = [years_known[i] for i, pop in enumerate(population_data) if pop > 0]
    valid_population = [pop for pop in population_data if pop > 0]

    extrapolated_population = np.nan
    coefficients = None

    if len(valid_years) >= model.degree + 1:
        try:
            coefficients = np.polyfit(valid_years, valid_population, model.degree)
            extrapolated_population = np.polyval(coefficients, model.x_future)
        except Exception as e:
            print(f"Помилка polyfit для {model.test_country}: {e}")
            extrapolated_population = np.nan
    else:
        extrapolated_population = np.nan

    years_for_plot = valid_years[:]
    populations_for_plot = valid_population[:]

    years_extended = years_for_plot + [model.x_future]
    populations_extended = populations_for_plot + [extrapolated_population]

    result = {
        'Country/Territory': model.test_country,
        'Extrapolated Population': extrapolated_population
    }

    if f'{model.x_future} Population' in country_row:
        actual = country_row[f'{model.x_future} Population']
        if not pd.isna(actual) and actual != 0:
            result['Actual Population'] = actual
            result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
            result['Percentage Error'] = (abs(extrapolated_population - actual) / actual * 100) if not pd.isna(extrapolated_population) and not pd.isna(actual) else np.nan
        else:
            result['Actual Population'] = actual
            result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
            result['Percentage Error'] = np.inf if actual == 0 and not pd.isna(extrapolated_population) else np.nan

    return result, (years_extended, populations_extended)