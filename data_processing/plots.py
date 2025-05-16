# plots.py
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import seaborn as sns # Для теплової карти


def y_axis_formatter(x, pos, normalize_plot=False):
    if normalize_plot:
        return f'{x:.2f}'
    else:
        if x >= 1e9:
            return f'{x*1e-9:.1f} млрд'
        elif x >= 1e6:
            return f'{x*1e-6:.0f} млн'
        elif x >= 1e3:
            return f'{x*1e-3:.0f} тис'
        return f'{x:.0f}'


def plot_country_population(ax, years, populations, country_name, normalize_plot=False, scaler_info=None):
    ax.clear() # Очищаємо попередній графік

    if years.size > 0 and not np.all(np.isnan(populations)):

        # Розділяємо дані на історичні та прогнозовану точку (остання точка - прогноз)
        historical_plot_years = years[:-1]
        historical_plot_populations = populations[:-1]
        forecast_year = years[-1]
        forecast_population = populations[-1]

        # Фільтруємо NaN значення для малювання історичних даних
        valid_history_indices = [i for i, pop in enumerate(historical_plot_populations) if not pd.isna(pop)]
        valid_historical_plot_years = historical_plot_years[valid_history_indices]
        valid_historical_plot_populations = historical_plot_populations[valid_history_indices]


        if valid_historical_plot_years.size > 0: # Малюємо історичні дані, якщо вони є
            ax.plot(valid_historical_plot_years, valid_historical_plot_populations, marker='o', linestyle='-', label='Історичні дані', color='skyblue')

        if not pd.isna(forecast_population): # Малюємо прогнозовану точку, якщо вона є
            ax.plot(forecast_year, forecast_population, marker='o', color='red', label='Прогноз', markersize=8)

        # Налаштування вісі Y
        all_valid_plot_values = populations[~np.isnan(populations)]
        y_min = np.min(all_valid_plot_values) if all_valid_plot_values.size > 0 else 0
        y_max = np.max(all_valid_plot_values) if all_valid_plot_values.size > 0 else 1 # Дефолтне значення, якщо даних немає


        # Визначення кроку та меж для вісі Y
        if normalize_plot:
             y_min_norm = np.min(populations) if populations[~np.isnan(populations)].size > 0 else -1
             y_max_norm = np.max(populations) if populations[~np.isnan(populations)].size > 0 else 1
             tick_step = (y_max_norm - y_min_norm) / 5 if (y_max_norm - y_min_norm) > 0 else 1
             y_ticks = np.linspace(y_min_norm, y_max_norm, 5) # 5 позначок для нормалізованих даних
             y_lim_min, y_lim_max = y_min_norm, y_max_norm
             y_label = f'Нормалізоване населення ({scaler_info[1]})' if scaler_info else 'Нормалізоване населення'
        else:
             y_label = 'Населення'
             # Логіка для визначення кроку для сирих даних
             tick_step = (y_max - y_min) / 5
             if tick_step > 0:
                 power = np.floor(np.log10(tick_step))
                 tick_step = np.ceil(tick_step / (10**power)) * (10**power)
             else:
                 tick_step = 1_000_000 # Дефолтний крок, якщо дані нульові або однакові

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
                 tick_start = y_min * 0.9 if y_min > 0 else -1 # Невеликий діапазон для однакових ненульових значень
                 tick_end = y_max * 1.1 if y_max > 0 else 1 # Невеликий діапазон
                 tick_step = (tick_end - tick_start) / 5 if (tick_end - tick_start) > 0 else 1

             y_ticks = np.arange(tick_start, tick_end + tick_step/2, tick_step)
             y_lim_min, y_lim_max = tick_start, tick_end # + (tick_step if not normalize_plot else 0)) # Додаємо невеликий відступ зверху


        ax.set(
            yticks=y_ticks,
            ylabel=y_label,
            ylim=(y_lim_min, y_lim_max),
            title=f'Динаміка населення для {country_name}'
        )

        # Налаштування вісі X
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(year)) for year in years])


        # Застосовуємо форматер
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: y_axis_formatter(x, pos, normalize_plot)))


        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    else:
        ax.text(0.5, 0.5, "Недостатньо даних для побудови графіка.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Динаміка населення для {country_name}')
        ax.set_xticks([]) # Прибираємо мітки, якщо немає даних
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Коригуємо розмір графіка, щоб уникнути обрізання міток


def plot_heatmap(ax, dataframe, columns_to_include):
    """
    Plots a correlation heatmap for specified numerical columns in a DataFrame on a given Axes object.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        dataframe (pd.DataFrame): The input DataFrame.
        columns_to_include (list): A list of column names to include in the heatmap.
    """
    ax.clear() # Очищаємо попередній графік

    # Вибираємо тільки числові стовпці з тих, що вказані для включення
    numeric_cols = dataframe[columns_to_include].select_dtypes(include=np.number)

    if numeric_cols.empty or numeric_cols.shape[1] < 2:
         ax.text(0.5, 0.5, "Недостатньо числових стовпців для теплової карти.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
         ax.set_title('Теплова карта кореляцій')
         ax.set_xticks([])
         ax.set_yticks([])
         return

    # Розраховуємо матрицю кореляції
    correlation_matrix = numeric_cols.corr()

    if correlation_matrix.empty:
         ax.text(0.5, 0.5, "Не вдалося розрахувати матрицю кореляції.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
         ax.set_title('Теплова карта кореляцій')
         ax.set_xticks([])
         ax.set_yticks([])
         return

    # Малюємо теплову карту
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)

    ax.set_title('Теплова карта кореляцій')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Коригуємо розмір, якщо потрібно