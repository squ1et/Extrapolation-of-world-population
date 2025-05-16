import numpy as np
import pandas as pd
import tkinter as tk

from extrapolation_method.model import PolynomialExtrapolationModel
from extrapolation_method.execute_model import execute_country_forecast

from data_processing import initial_analysis
from data_processing import preprocesing
from data_processing import plots

def update_years(year_var, year_menu, full_years, known_years, error_check_var):
    if error_check_var.get():
        year_menu['values'] = known_years
        if year_var.get() not in known_years:
            year_var.set(2022)
    else:
        year_menu['values'] = full_years
        if year_var.get() not in full_years:
             if year_var.get() < 2071:
                  year_var.set(2025)
    year_menu.set(year_var.get())


def on_submit(country_var, year_var, error_check_var, normalize_plot_var, model, output_text, ax_left, ax_right, canvas, heatmap_cols):
    selected_country = country_var.get()
    try:
        selected_year = int(year_var.get())
    except ValueError:
        output_text.insert(tk.END, "Enter real year\n")
        return

    estimate_error = error_check_var.get()
    normalize_plot = normalize_plot_var.get()

    model.test_country = selected_country
    model.x_future = selected_year

    output_text.delete(1.0, tk.END)

    result, plot_data = execute_country_forecast(model)

    # --- Підготовка даних для графіка населення ---
    plot_years, populations_for_plot = plot_data
    plot_populations = np.array(populations_for_plot) # Працюємо з numpy масивом
    plot_years = np.array(plot_years)

    scaler_info = None # Для збереження інформації про скейлер, якщо нормалізація відбулася

    if normalize_plot:
        # Застосовуємо нормалізацію до даних для графіка (історичні + прогноз)
        # normalize_country_population приймає список/масив і повертає нормалізований масив
        normalized_plot_populations, scaler_info = preprocesing.normalize_country_population(plot_populations)
        # Перевіряємо, чи нормалізація була успішною
        if scaler_info and scaler_info[0] is not None:
             plot_populations = normalized_plot_populations
             # scaler_info вже містить метод нормалізації
             output_text.insert(tk.END, f"Дані на графіку нормалізовано ({scaler_info[1]}).\n")
        else:
             # Якщо нормалізація не вдалася, виводимо повідомлення
             output_text.insert(tk.END, "Не вдалося нормалізувати дані для графіка. Відображено сирі дані.\n")
             normalize_plot = False # Вимикаємо флаг нормалізації графіка, якщо вона не вдалася

    plots.plot_country_population(ax_left, plot_years, plot_populations, selected_country, normalize_plot, scaler_info)

    # --- Малюємо теплову карту за допомогою функції з plots.py ---
    # Теплова карта будується на основі всього датасету, не лише вибраної країни
    plots.plot_heatmap(ax_right, model.df, heatmap_cols)

    # --- Оновлюємо GUI вивід ---
    if result:
        output_text.insert(tk.END, f"Країна: {result.get('Country/Territory', 'N/A')}\n")
        output_text.insert(tk.END, f"Рік для прогнозу/перевірки: {selected_year}\n")

        predicted_pop = result.get('Extrapolated Population', np.nan)

        # Якщо дані були нормалізовані для графіка, прогнозоване значення в текстовому виводі має бути сирим
        # (якщо ви хочете виводити сире значення) або також нормалізованим (якщо ви хочете послідовність)
        # Наразі виводимо сире, як і раніше.
        if not pd.isna(predicted_pop):
            predicted_pop_formatted = f"{predicted_pop:,.0f}".replace(",", " ")
            output_text.insert(tk.END, f"Прогнозоване населення: {predicted_pop_formatted}\n")
        else:
            output_text.insert(tk.END, "Прогнозоване населення: Недостатньо даних або помилка розрахунку\n")

        # Вивід похибки (логіка залишається без змін)
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
    else:
         output_text.insert(tk.END, f"Дані для країни '{selected_country}' не знайдені або сталася помилка при розрахунку.\n")


    canvas.draw() # Оновлюємо Tkinter Canvas, який містить обидва графіки

# display_statistics та perform_column_normalization залишаються тут або переносяться за потребою
# (в поточному плані вони в app_logic)

def display_statistics(column_name, df, output_widget):
    """Calculates and displays statistics for a given column (column-wise)."""
    output_widget.delete(1.0, tk.END)
    stats, error = initial_analysis.calculate_statistics(df, column_name)

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


def perform_column_normalization(column_name, df, output_widget, method='minmax'):
     """Performs column-wise data normalization and displays a confirmation message."""
     output_widget.insert(tk.END, f"\nСпроба нормалізації стовпця '{column_name}' методом '{method}'...\n")
     normalized_df = preprocesing.normalize_column(df, column_name, method=method)

     normalized_col_name = f'{column_name}_Normalized_{method}'
     if normalized_col_name in normalized_df.columns:
          output_widget.insert(tk.END, f"Нормалізація '{column_name}' методом '{method}' успішна. Додано стовпець '{normalized_col_name}'.\n")
          try:
              sample_data = normalized_df[[column_name, normalized_col_name]].head().to_string()
              output_widget.insert(tk.END, "\nПриклад нормалізованих даних:\n")
              output_widget.insert(tk.END, sample_data + "\n")
          except Exception as e:
               output_widget.insert(tk.END, f"Не вдалося вивести приклад даних: {e}\n")

     else:
          output_widget.insert(tk.END, f"Нормалізація '{column_name}' методом '{method}' не вдалася.\n")
          if normalized_df is df:
               output_widget.insert(tk.END, "Причина: Можливо, помилка в процесі нормалізації або стовпець не містить числових даних.\n")