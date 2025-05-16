import kagglehub
import pandas as pd

def load_dataset():
    try:
        path_to_data = kagglehub.dataset_download("iamsouravbanerjee/world-population-dataset")
        data = pd.read_csv(f'{path_to_data}/data.csv')
        print("Successfully downloaded")
    except Exception as e:
        print(f"[ERROR]: {e}")
        print("Trying to load local data")
        try:
            data = pd.read_csv('data.csv')
            print("Loaded local data")
        except FileNotFoundError:
            print("data.csv not found")
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

    return data