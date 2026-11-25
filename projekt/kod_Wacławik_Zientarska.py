import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("Telco.csv")

# Podstawowe informacje na temat danych
print(data.head())
print(data.shape)
print(data.describe())
data.info()

# Zbiór danych zawiera 7043 obserwacje (klientów) i 21 zmiennych.
# Zmienna customerID jest identyfikatorem klienta i nie niesie informacji predykcyjnej.
# Zmienną objaśnianą jest Churn (czy klient odszedł: Yes/No).
# Wśród zmiennych objaśniających są zmienne liczbowe (tenure, MonthlyCharges, TotalCharges),
# zmienna binarna zapisana jako 0/1 (SeniorCitizen) oraz zmienne kategoryczne zapisane tekstowo.

# Funkcja describe() pokazuje tylko trzy zmienne, ponieważ TotalCharges została wczytana jako tekst (typ object),
# mimo że zawiera kwoty. Wskazuje to na nieliczbowe wartości w tej kolumnie.

# Pracujemy na kopii danych, aby móc w razie potrzeby wrócić do oryginału.

dane = data.copy()
dane = dane.drop(columns="customerID")

dane["TotalCharges"] = pd.to_numeric(dane["TotalCharges"], errors="coerce")
# Po zamianie TotalCharges na typ liczbowy pojawiło się 11 braków danych.
# W pliku były to puste wartości (spacje), więc isna() nie wykrywało ich na początku
braki = dane.isna().sum()
print(braki > 0)

print(dane.loc[dane["TotalCharges"].isna(), ["tenure", "MonthlyCharges", "TotalCharges"]])
# Wszystkie 11 obserwacji ma tenure = 0, czyli są to nowi klienci, którzy nie zostali jeszcze obciążeni żadną opłatą.
# Jest to również istotna informacja, dlatego brak zamieniany jest na 0.
dane["TotalCharges"] = dane["TotalCharges"].fillna(0)

braki = dane.isna().sum()
print(braki > 0)
# Po tej operacji w zbiorze nie ma braków danych.

dane.info()

# Weryfikacja wartości dostających
kolumny_liczbowe = ["tenure", "MonthlyCharges", "TotalCharges"]

fig, osie = plt.subplots(1, 3, figsize=(14, 4))
for os_, kolumna in zip(osie, kolumny_liczbowe):
    dane.boxplot(column=kolumna, by="Churn", ax=os_)
    os_.set_title(kolumna)
plt.suptitle("")
plt.tight_layout()
plt.show()

for kolumna in ["tenure", "TotalCharges"]:
    for grupa, wartosci in dane.groupby("Churn")[kolumna]:
        q1, q3 = wartosci.quantile([0.25, 0.75])
        iqr = q3 - q1
        dolna, gorna = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        liczba = ((wartosci < dolna) | (wartosci > gorna)).sum()
        print(f"{kolumna}, Churn = {grupa}: Q1 = {q1:.1f}, Q3 = {q3:.1f}, "
              f"górna granica = {gorna:.1f}, wartości odstające: {liczba}")

odeszli = dane[dane["Churn"] == "Yes"]
q1, q3 = odeszli["tenure"].quantile([0.25, 0.75])
odstajacy = odeszli[odeszli["tenure"] > q3 + 1.5 * (q3 - q1)]

print("Liczba klientów:", len(odstajacy))
print(odstajacy[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(1))
print(odstajacy["Contract"].value_counts())

# Na wykresach pudełkowych w podziale na grupy widoczne są punkty odstające wśród klientów, którzy odeszli:
# 23 dla tenure (górna granica 69,5 miesiąca) i 109 dla TotalCharges (górna granica 5626,5). Wynikają one z tego,
# że w tej grupie większość klientów odchodzi w pierwszych miesiącach (Q1 = 2, Q3 = 29),
# więc długi staż jest w niej nietypowy.

# W grupie klientów, którzy zostali, nie ma wartości odstających.

# 23 klientów odstających pod względem stażu to klienci z bardzo długim stażem (70–72 miesiące),
# głównie z umowami dwuletnimi (14) i rocznymi (7), płacący wysokie opłaty miesięczne (średnio 96,8).
# Są to wartości realne, a nie błędy danych.

for kolumna in kolumny_liczbowe:
    q1, q3 = dane[kolumna].quantile([0.25, 0.75])
    iqr = q3 - q1
    liczba = ((dane[kolumna] < q1 - 1.5 * iqr) | (dane[kolumna] > q3 + 1.5 * iqr)).sum()
    print(f"{kolumna}: {liczba} wartości odstających")

# W całym zbiorze, według reguły 1,5·IQR, nie ma wartości odstających w żadnej zmiennej liczbowej.
# Punkty widoczne na wykresach są odstające tylko w obrębie jednej grupy.
# Nie usuwamy żadnych obserwacji: są to rzeczywiste przypadki klientów, którzy odeszli mimo długiego stażu,
# a model powinien się na nich uczyć.

print(dane[kolumny_liczbowe].corr().round(3))
# Zmienna TotalCharges jest silnie skorelowana z tenure (0,826) i MonthlyCharges (0,651).
# Jej korelacja z iloczynem tenure · MonthlyCharges wynosi 0,9996,
# czyli łączna kwota jest praktycznie równa stażowi pomnożonemu przez opłatę miesięczną.
print("Korelacja TotalCharges z tenure*MonthlyCharges:",
      np.corrcoef(dane["tenure"] * dane["MonthlyCharges"], dane["TotalCharges"])[0, 1].round(4))

# Zmienna nie wnosi więc nowej informacji, a jej pozostawienie mogłoby zaburzyć interpretację modeli
# (wpływ stażu zostałby rozdzielony między dwie zmienne). Usuwamy TotalCharges ze zbioru używanego do modelowania.
dane = dane.drop(columns="TotalCharges")

# Zmienne binarne - encoding

uslugi_internetowe = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
dane[uslugi_internetowe] = dane[uslugi_internetowe].replace("No internet service", "No")
dane["MultipleLines"] = dane["MultipleLines"].replace("No phone service", "No")

# W zmiennych opisujących usługi internetowe (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
# StreamingMovies) kategoria „No internet service” została zamieniona na „No”, a w MultipleLines kategoria
# „No phone service” na „No”. Informacja o braku internetu lub telefonu jest już zawarta w zmiennych InternetService
# i PhoneService, więc nie zostaje utracona. Dzięki temu siedem zmiennych stało się zmiennymi binarnymi.

binarne = ["Partner", "Dependents", "PhoneService", "MultipleLines", "PaperlessBilling", "Churn"] + uslugi_internetowe
for kolumna in binarne:
    dane[kolumna] = dane[kolumna].map({"Yes": 1, "No": 0})
dane["gender"] = dane["gender"].map({"Male": 1, "Female": 0})

dane = pd.get_dummies(dane, columns=["InternetService", "Contract", "PaymentMethod"], dtype=int)

print(dane.shape)
print("Braki:", dane.isna().sum().sum())
dane.info()

# Zmienne binarne zakodowano jako 0/1, a zmienne kategoryczne o większej liczbie kategorii (InternetService, Contract,
# PaymentMethod) zamieniono na zmienne zero-jedynkowe. Kodowanie wykonano przed podziałem na zbiór uczący i testowy,
# ponieważ jest to przekształcenie deterministyczne, które nie wyznacza żadnych parametrów na podstawie danych.

# Przygotowany zbiór zawiera 7043 obserwacje i 26 kolumn: 25 zmiennych objaśniających oraz zmienną Churn.
# Wszystkie zmienne są liczbowe i nie ma braków danych, więc zbiór jest gotowy do podziału na zbiór uczący i testowy.


# EDA - Kasia
# Podział na zbiór uczący i testowy
# Opracowanie zmiennych katgorycznych

