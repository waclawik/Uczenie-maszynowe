import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data = pd.read_csv("Telco.csv", decimal=".", sep=",")
print(data.head())

print(data.shape)
data.info()
print(data.describe())

braki = data.isna().sum()
print(braki[braki > 0])

# Nie ma braków danych
# Do zrobienia, jeśli chodzi o dane - Iza: uporządkowanie typów danych przed EDA
# EDA - Kasia
# Podział na zbiór uczący i testowy
# Opracowanie zmiennych katgorycznych

