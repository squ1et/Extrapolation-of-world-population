import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from extrapolation_method.model import PolynomialExtrapolationModel

from data_processing.preprocesing import process_data

from app_logic import (
    update_years,
    on_submit,
    display_statistics,
    perform_column_normalization
)

import tkinter as tk
from tkinter import ttk

data = process_data()

model = PolynomialExtrapolationModel(data, x_future=2025)

heatmap_columns = [col for col in model.df.columns if 'Population' in col and col != 'World Population Percentage']
heatmap_columns.extend(['Area (Km²)', 'Density (per Km²)', 'Growth Rate', 'World Population Percentage'])
heatmap_columns = [col for col in heatmap_columns if col in model.df.columns] # Фільтруємо наявні


def launch_gui(model):
    root = tk.Tk()
    root.title("Аналіз та Прогнозування Населення")
    root.geometry("1200x850")

    main_canvas = tk.Canvas(root)
    main_scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
    main_canvas.configure(yscrollcommand=main_scrollbar.set)

    main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
    main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollable_frame = ttk.Frame(main_canvas)

    frame_window_id = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")


    def on_canvas_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        canvas_width = main_canvas.winfo_width()
        main_canvas.itemconfig(frame_window_id, width=canvas_width)


    main_canvas.bind("<Configure>", on_canvas_configure)
    def on_mousewheel(event):
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    main_canvas.bind_all("<MouseWheel>", on_mousewheel)
    main_canvas.bind_all("<Button-4>", on_mousewheel)
    main_canvas.bind_all("<Button-5>", on_mousewheel)


    predict_frame = ttk.LabelFrame(scrollable_frame, text="Прогнозування Населення")
    predict_frame.pack(pady=10, padx=10, fill="x", expand=False)


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
    error_check = tk.Checkbutton(predict_controls_frame, text="Перевірити наявний рік (для похибки)", variable=error_check_var, command=lambda: update_years(year_var, year_menu, full_years, known_years, error_check_var))
    error_check.grid(row=2, column=1, sticky='w', padx=5, pady=5)

    normalize_plot_var = tk.BooleanVar()
    normalize_plot_check = tk.Checkbutton(predict_controls_frame, text="Нормалізувати дані на графіку", variable=normalize_plot_var)
    normalize_plot_check.grid(row=2, column=2, sticky='w', padx=5, pady=5)


    update_years(year_var, year_menu, full_years, known_years, error_check_var)


    predict_button = tk.Button(predict_controls_frame, text="Прогнозувати", command=lambda: on_submit(country_var, year_var, error_check_var, normalize_plot_var, model, output_text, ax_left, ax_right, canvas, heatmap_columns)) # Передаємо heatmap_columns
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

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

    stats_button = tk.Button(stats_controls_frame, text="Розрахувати статистику", command=lambda: display_statistics(stats_column_var.get(), model.df, stats_output_text))
    stats_button.grid(row=0, column=2, padx=10, pady=5)

    stats_output_container = tk.Frame(stats_frame)
    stats_output_container.pack(pady=5, padx=10, fill="x", expand=True)

    stats_output_text = tk.Text(stats_output_container, height=10, width=90)
    stats_output_text.pack(side=tk.LEFT, fill="x", expand=True)

    stats_scrollbar = ttk.Scrollbar(stats_output_container, orient=tk.VERTICAL, command=stats_output_text.yview)
    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    stats_output_text['yscrollcommand'] = stats_scrollbar.set


    root.mainloop()


if __name__ == "__main__":
    data = process_data()
    model = PolynomialExtrapolationModel(data, x_future=2025)

    launch_gui(model)