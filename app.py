import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror, showinfo


try:
    path_to_data = kagglehub.dataset_download("iamsouravbanerjee/world-population-dataset")
    data = pd.read_csv(f'{path_to_data}/data.csv')
    print("Датасет завантажено з Kaggle Hub.")
except Exception as e:
    print(f"Помилка завантаження датасету з Kaggle Hub: {e}")
    print("Спроба завантажити локально (якщо файл є) або використовувати фіктивні дані.")
    try:
        data = pd.read_csv('data.csv')
        print("Датасет завантажено локально.")
    except FileNotFoundError:
        print("Локальний файл data.csv не знайдено. Створення фіктивних даних для демонстрації.")
        dummy_data = {
            'Country/Territory': ['Україна', 'Польща', 'Німеччина', 'Франція', 'Італія', 'Іспанія', 'Канада', 'Австралія', 'Бразилія', 'Індія', 'Китай', 'США'],
            '1970 Population': [40e6, 30e6, 70e6, 50e6, 50e6, 34e6, 21e6, 12e6, 95e6, 550e6, 800e6, 200e6],
            '1980 Population': [42e6, 35e6, 75e6, 55e6, 55e6, 37e6, 24e6, 14e6, 120e6, 690e6, 980e6, 225e6],
            '1990 Population': [45e6, 38e6, 80e6, 60e6, 58e6, 39e6, 28e6, 17e6, 150e6, 870e6, 1150e6, 250e6],
            '2000 Population': [48e6, 40e6, 82e6, 62e6, 60e6, 40e6, 30e6, 19e6, 175e6, 1050e6, 1270e6, 280e6],
            '2010 Population': [46e6, 38e6, 81e6, 64e6, 60e6, 46e6, 34e6, 22e6, 195e6, 1230e6, 1340e6, 308e6],
            '2015 Population': [44e6, 38e6, 82e6, 65e6, 60.5e6, 46.5e6, 35e6, 24e6, 205e6, 1310e6, 1390e6, 320e6],
            '2020 Population': [41e6, 38e6, 83e6, 65.5e6, 60.3e6, 47e6, 37e6, 25.5e6, 212e6, 1380e6, 1420e6, 331e6],
            '2022 Population': [39e6, 38e6, 83.2e6, 65.8e6, 60.2e6, 47.3e6, 38e6, 26e6, 214e6, 1400e6, 1425e6, 334e6],
            'Area (Km²)': [603628, 312696, 357022, 640679, 301340, 505992, 9984670, 7741220, 8515767, 3287590, 9640011, 9372610],
            'Density (per Km²)': [None] * 12, 'Growth Rate': [0.99] * 12, 'World Population Percentage': [None] * 12
        }
        data = pd.DataFrame(dummy_data)

data.fillna(0, inplace=True)


class PolynomialExtrapolationModel:
    def __init__(self, df, x_future, degree=2, test_year=2022, test_country="Ukraine"):
        self.df = df
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


    def test_model_by_country(self):
        country_row = self.df[self.df['Country/Territory'] == self.test_country]

        if country_row.empty:
            return None, None

        country_row = country_row.iloc[0]
        years_known = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
        population_data = [country_row[f'{year} Population'] for year in years_known]

        valid_years = [years_known[i] for i, pop in enumerate(population_data) if pop > 0]
        valid_population = [pop for pop in population_data if pop > 0]

        extrapolated_population = np.nan
        coefficients = None

        if len(valid_years) >= self.degree + 1:
            try:
                coefficients = np.polyfit(valid_years, valid_population, self.degree)
                extrapolated_population = np.polyval(coefficients, self.x_future)
            except Exception as e:
                print(f"Помилка polyfit для {self.test_country}: {e}")
                extrapolated_population = np.nan
        else:
            extrapolated_population = np.nan

        years_for_plot = valid_years[:]
        populations_for_plot = valid_population[:]

        years_extended = years_for_plot + [self.x_future]
        populations_extended = populations_for_plot + [extrapolated_population]

        result = {
            'Country/Territory': self.test_country,
            'Extrapolated Population': extrapolated_population
        }

        if f'{self.x_future} Population' in country_row:
            actual = country_row[f'{self.x_future} Population']
            if not pd.isna(actual) and actual != 0:
                result['Actual Population'] = actual
                result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
                result['Percentage Error'] = (abs(extrapolated_population - actual) / actual * 100) if not pd.isna(extrapolated_population) and not pd.isna(actual) else np.nan
            else:
                result['Actual Population'] = actual
                result['Absolute Error'] = abs(extrapolated_population - actual) if not pd.isna(extrapolated_population) else np.nan
                result['Percentage Error'] = np.inf if actual == 0 and not pd.isna(extrapolated_population) else np.nan

        return result, (years_extended, populations_extended)


    def calculate_statistics(self, column_name):
        if column_name not in self.df.columns:
            return None, "Стовпець не знайдено."
        column_data = self.df[column_name]

        if column_data.empty:
            return None, "Стовпець порожній або містить лише NaN/0."

        try:
            numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()

            if numeric_data.empty:
                return None, "Стовпець не містить числових даних (окрім NaN/0)."

            mean = numeric_data.mean()
            median = numeric_data.median()
            variance = numeric_data.var()
            std_dev = numeric_data.std()
            maximum = numeric_data.max()
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
            return None, f"Помилка при розрахунку статистики: {e}"

    def normalize_data(self, column_name, method='minmax'):
        if column_name not in self.df.columns:
            print(f"Помилка нормалізації: Стовпець '{column_name}' не знайдено.")
            return self.df.copy()

        df_normalized = self.df.copy()

        col_to_normalize = pd.to_numeric(df_normalized[column_name], errors='coerce')

        valid_indices = col_to_normalize.dropna().index
        data_for_scaling = col_to_normalize.loc[valid_indices].values.reshape(-1, 1)

        if data_for_scaling.size == 0:
            print(f"Помилка нормалізації: Стовпець '{column_name}' не містить числових даних для нормалізації.")
            return self.df.copy()

        try:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:
                print(f"Помилка нормалізації: Невідомий метод '{method}'. Використано 'minmax'.")
                scaler = MinMaxScaler()

            normalized_data = scaler.fit_transform(data_for_scaling)

            normalized_series = pd.Series(normalized_data.flatten(), index=valid_indices, name=f'{column_name}_Normalized_{method}')
            df_normalized = df_normalized.join(normalized_series)

            print(f"Стовпець '{column_name}' успішно нормалізовано методом '{method}'.")
            return df_normalized

        except Exception as e:
            print(f"Помилка нормалізації для стовпця '{column_name}' методом '{method}': {e}")
            return self.df.copy()


def launch_gui(model):
    root = tk.Tk()
    root.title("Аналіз та Прогнозування Населення")
    root.geometry("1200x850")

    # Налаштування Canvas та Scrollbar для прокручування всього вікна
    main_canvas = tk.Canvas(root)
    main_scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
    main_canvas.configure(yscrollcommand=main_scrollbar.set)

    main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Фрейм, який буде містити весь вміст вікна
    scrollable_frame = ttk.Frame(main_canvas)

    # Додавання фрейму з вмістом до Canvas
    # Параметр width=1200 гарантує, що фрейм має ширину вікна і не потрібна горизонтальна прокрутка
    # anchor='nw' фіксує фрейм у верхньому лівому куті canvas
    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1200)


    # Функція для оновлення області прокручування Canvas
    def on_frame_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))

    # Прив'язка функції до події зміни розміру фрейму
    scrollable_frame.bind("<Configure>", on_frame_configure)


    # --- Фрейм для прогнозування (перенесено у scrollable_frame) ---
    predict_frame = ttk.LabelFrame(scrollable_frame, text="Прогнозування Населення")
    predict_frame.pack(pady=10, padx=10, fill="x")

    predict_controls_frame = tk.Frame(predict_frame)
    predict_controls_frame.pack(pady=5)

    tk.Label(predict_controls_frame, text="Оберіть країну:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    country_var = tk.StringVar(value=model.test_country)
    countries = sorted(model.df['Country/Territory'].unique().tolist())
    country_menu = ttk.Combobox(predict_controls_frame, textvariable=country_var, values=countries, state='readonly', width=30)
    country_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')

    tk.Label(predict_controls_frame, text="Оберіть рік для прогнозу/перевірки:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    year_var = tk.IntVar(value=model.test_year)
    full_years = list(range(1970, 2071))
    known_years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
    year_menu = ttk.Combobox(predict_controls_frame, textvariable=year_var, values=full_years, state='readonly', width=10)
    year_menu.grid(row=1, column=1, padx=5, pady=5, sticky='w')

    error_check_var = tk.BooleanVar()
    error_check = tk.Checkbutton(predict_controls_frame, text="Перевірити наявний рік (для похибки)", variable=error_check_var, command=lambda: update_years(year_var, year_menu, full_years, known_years))
    error_check.grid(row=2, column=1, sticky='w', padx=5, pady=5)

    def update_years(year_var, year_menu, full_years, known_years):
        if error_check_var.get():
            year_menu['values'] = known_years
            if year_var.get() not in known_years:
                year_var.set(2022)
        else:
            year_menu['values'] = full_years
        year_menu.set(year_var.get())

    update_years(year_var, year_menu, full_years, known_years)

    predict_button = tk.Button(predict_controls_frame, text="Прогнозувати", command=lambda: on_submit(country_var, year_var, error_check_var, model, output_text, ax_left, canvas))
    predict_button.grid(row=3, column=1, pady=10, padx=5, sticky='w')

    predict_output_container = tk.Frame(predict_frame)
    predict_output_container.pack(pady=5, padx=10, fill="x", expand=True)

    output_text = tk.Text(predict_output_container, height=8, width=90)
    output_text.pack(side=tk.LEFT, fill="x", expand=True)

    predict_scrollbar = ttk.Scrollbar(predict_output_container, orient=tk.VERTICAL, command=output_text.yview)
    predict_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    output_text['yscrollcommand'] = predict_scrollbar.set


    plot_frame = tk.Frame(predict_frame)
    plot_frame.pack(pady=10, expand=True, fill=tk.BOTH)

    fig, ax_left = plt.subplots(1, 1, figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def on_submit(country_var, year_var, error_check_var, model, output_text, ax_left, canvas):
        selected_country = country_var.get()
        try:
            selected_year = int(year_var.get())
        except ValueError:
            output_text.insert(tk.END, "Будь ласка, оберіть коректний рік.\n")
            return

        estimate_error = error_check_var.get()

        model.test_country = selected_country
        model.x_future = selected_year
        model.results = []

        output_text.delete(1.0, tk.END)
        ax_left.clear()

        result, plot_data = model.test_model_by_country()

        if result:
            output_text.insert(tk.END, f"Країна: {result.get('Country/Territory', 'N/A')}\n")
            output_text.insert(tk.END, f"Рік для прогнозу/перевірки: {selected_year}\n")

            predicted_pop = result.get('Extrapolated Population', np.nan)

            if not pd.isna(predicted_pop):
                predicted_pop_formatted = f"{predicted_pop:,.0f}".replace(",", " ")
                output_text.insert(tk.END, f"Прогнозоване населення: {predicted_pop_formatted}\n")
            else:
                output_text.insert(tk.END, "Прогнозоване населення: Недостатньо даних або помилка розрахунку\n")


            years_for_plot, populations_for_plot = plot_data

            if years_for_plot and not all(pd.isna(populations_for_plot)):
                historical_years = years_for_plot[:-1]
                historical_populations = populations_for_plot[:-1]

                valid_history_indices = [i for i, pop in enumerate(historical_populations) if not pd.isna(pop)]
                valid_historical_years = [historical_years[i] for i in valid_history_indices]
                valid_historical_populations = [historical_populations[i] for i in valid_history_indices]

                if valid_historical_years:
                    ax_left.plot(valid_historical_years, valid_historical_populations, marker='o', linestyle='-', label='Історичні дані', color='skyblue')

                if not pd.isna(predicted_pop):
                    ax_left.plot(years_for_plot[-1], populations_for_plot[-1], marker='o', color='red', label='Прогноз', markersize=8)

                all_population_values = [pop for pop in populations_for_plot if not pd.isna(pop)]
                y_min = min(all_population_values) if all_population_values else 0
                y_max = max(all_population_values) if all_population_values else 10e6

                tick_step = (y_max - y_min) / 5
                if tick_step > 0:
                    power = np.floor(np.log10(tick_step))
                    tick_step = np.ceil(tick_step / (10**power)) * (10**power)
                else:
                    tick_step = 1_000_000

                if tick_step < 1000: tick_step = 1000
                elif tick_step < 10000: tick_step = 10000
                elif tick_step < 100000: tick_step = 100000
                elif tick_step < 1000000: tick_step = 1000000
                elif tick_step < 10000000: tick_step = 5000000


                tick_start = int(np.floor(y_min / tick_step) * tick_step)
                tick_end = int(np.ceil(y_max / tick_step) * tick_step)

                if tick_end <= tick_start and y_max > y_min:
                    tick_end = tick_start + tick_step
                elif tick_end <= tick_start and y_max == y_min:
                    tick_start = y_min * 0.9 if y_min > 0 else 0
                    tick_end = y_max * 1.1 if y_max > 0 else tick_step
                    tick_step = (tick_end - tick_start) / 5

                y_ticks = np.arange(tick_start, tick_end + tick_step/2, tick_step)

                ax_left.plot(years_for_plot, populations_for_plot, linestyle='None')

                ax_left.set(
                    yticks=y_ticks,
                    ylabel='Населення',
                    ylim=(tick_start, tick_end),
                    title=f'Динаміка населення для {selected_country}'
                )

                ax_left.set_xticks(years_for_plot)
                ax_left.set_xticklabels([str(int(year)) for year in years_for_plot])

                def millions_formatter(x, pos):
                    if x >= 1e9:
                        return f'{x*1e-9:.1f} млрд'
                    elif x >= 1e6:
                        return f'{x*1e-6:.0f} млн'
                    elif x >= 1e3:
                        return f'{x*1e-3:.0f} тис'
                    return f'{x:.0f}'
                ax_left.yaxis.set_major_formatter(FuncFormatter(millions_formatter))


                ax_left.legend(loc='upper left')
                ax_left.grid(True, linestyle='--', alpha=0.6)

                plt.setp(ax_left.get_xticklabels(), rotation=45, ha="right")


            else:
                output_text.insert(tk.END, "Недостатньо даних для побудови графіка.\n")

            if estimate_error and 'Actual Population' in result:
                actual_pop = result.get('Actual Population', np.nan)
                abs_error = result.get('Absolute Error', np.nan)
                perc_error = result.get('Percentage Error', np.nan)

                if not pd.isna(actual_pop):
                    actual_pop_formatted = f"{actual_pop:,.0f}".replace(",", " ")
                    output_text.insert(tk.END, f"Фактичне населення ({selected_year}): {actual_pop_formatted}\n")
                else:
                    output_text.insert(tk.END, f"Фактичне населення ({selected_year}): Невідомо\n")

                if not pd.isna(abs_error):
                    abs_error_formatted = f"{abs_error:,.0f}".replace(",", " ")
                    output_text.insert(tk.END, f"Похибка (осіб): {abs_error_formatted}\n")
                else:
                    output_text.insert(tk.END, "Похибка (осіб): N/A\n")

                if not pd.isna(perc_error):
                    if perc_error == np.inf:
                        output_text.insert(tk.END, "Похибка (%): Нескінченність (фактичне населення 0)\n")
                    else:
                        output_text.insert(tk.END, f"Похибка (%): {perc_error:.2f}%\n")
                else:
                    output_text.insert(tk.END, "Похибка (%): N/A\n")

                if (pd.isna(actual_pop) or actual_pop == 0) and (not pd.isna(predicted_pop) or predicted_pop != 0):
                    output_text.insert(tk.END, "Примітка: Похибка % може бути нескінченною або невизначеною, якщо фактичне населення 0 або невідоме.\n")


            canvas.draw()
            plt.tight_layout(rect=[0, 0, 1, 0.95])


        else:
            output_text.insert(tk.END, f"Дані для країни '{selected_country}' не знайдені або сталася помилка при розрахунку.\n")
            canvas.draw()
            plt.tight_layout(rect=[0, 0, 1, 0.95])


    # --- Фрейм для статистичного аналізу (перенесено у scrollable_frame) ---
    stats_frame = ttk.LabelFrame(scrollable_frame, text="Статистичний Аналіз Даних")
    stats_frame.pack(pady=10, padx=10, fill="x")

    stats_controls_frame = tk.Frame(stats_frame)
    stats_controls_frame.pack(pady=5)

    tk.Label(stats_controls_frame, text="Оберіть стовпець для аналізу:").grid(row=0, column=0, padx=5, pady=5, sticky='e')

    statistical_columns = [col for col in model.df.columns if 'Population' in col and col != 'World Population Percentage']
    statistical_columns.extend(['Area (Km²)', 'Density (per Km²)', 'Growth Rate', 'World Population Percentage'])
    statistical_columns = [col for col in statistical_columns if col in model.df.columns]

    stats_column_var = tk.StringVar()
    stats_column_menu = ttk.Combobox(stats_controls_frame, textvariable=stats_column_var, values=statistical_columns, state='readonly', width=30)
    stats_column_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    if statistical_columns:
        stats_column_menu.set(statistical_columns[0])

    stats_button = tk.Button(stats_controls_frame, text="Розрахувати статистику", command=lambda: display_statistics(stats_column_var.get(), model, stats_output_text))
    stats_button.grid(row=0, column=2, padx=10, pady=5)

    tk.Button(stats_controls_frame, text="Нормалізувати (Min-Max)", command=lambda: perform_normalization(stats_column_var.get(), model, stats_output_text, method='minmax')).grid(row=0, column=3, padx=5, pady=5)
    tk.Button(stats_controls_frame, text="Нормалізувати (Z-score)", command=lambda: perform_normalization(stats_column_var.get(), model, stats_output_text, method='standard')).grid(row=0, column=4, padx=5, pady=5)

    # Контейнер для текстового виводу статистики та смуги прокручування
    stats_output_container = tk.Frame(stats_frame)
    stats_output_container.pack(pady=5, padx=10, fill="x", expand=True)

    stats_output_text = tk.Text(stats_output_container, height=10, width=90)
    stats_output_text.pack(side=tk.LEFT, fill="x", expand=True)

    stats_scrollbar = ttk.Scrollbar(stats_output_container, orient=tk.VERTICAL, command=stats_output_text.yview)
    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    stats_output_text['yscrollcommand'] = stats_scrollbar.set


    def display_statistics(column_name, model, output_widget):
        output_widget.delete(1.0, tk.END)
        stats, error = model.calculate_statistics(column_name)

        if error:
            output_widget.insert(tk.END, f"Помилка: {error}\n")
        elif stats:
            output_widget.insert(tk.END, f"Статистика для стовпця '{column_name}':\n\n")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:,.4f}".replace(",", " ")
                    if formatted_value.endswith('.0000'):
                        formatted_value = formatted_value[:-5]
                    output_widget.insert(tk.END, f"{key}: {formatted_value}\n")
                else:
                    output_widget.insert(tk.END, f"{key}: {value}\n")
        else:
            output_widget.insert(tk.END, f"Не вдалося розрахувати статистику для стовпця '{column_name}'. Перевірте дані.\n")

    def perform_normalization(column_name, model, output_widget, method='minmax'):
        output_widget.insert(tk.END, f"\nСпроба нормалізації стовпця '{column_name}' методом '{method}'...\n")
        normalized_df = model.normalize_data(column_name, method=method)
        if f'{column_name}_Normalized_{method}' in normalized_df.columns:
            output_widget.insert(tk.END, f"Нормалізація '{column_name}' методом '{method}' успішна.\n")
        else:
            output_widget.insert(tk.END, f"Нормалізація '{column_name}' методом '{method}' не вдалася.\n")


    root.mainloop()

model = PolynomialExtrapolationModel(data, x_future=2025)
launch_gui(model)