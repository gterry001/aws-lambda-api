import pandas as pd
import os

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
print("Contenido de /var/task/app:", os.listdir(BASE_DIR))


def run_analysis():
    portfolio = pd.read_excel(portfolio_path)
    return {
        "ordenes":"leido"
    }











