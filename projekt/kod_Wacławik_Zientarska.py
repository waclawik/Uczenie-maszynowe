import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, make_scorer, RocCurveDisplay

data = pd.read_csv("c:/Users/Kasia/Desktop/IiE/II stopień/Uczenie maszynowe/Uczenie maszynowe/Uczenie-maszynowe/projekt/Telco.csv")

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
print(braki[braki > 0])

print(dane.loc[dane["TotalCharges"].isna(), ["tenure", "MonthlyCharges", "TotalCharges"]])
# Wszystkie 11 obserwacji ma tenure = 0, czyli są to nowi klienci, którzy nie zostali jeszcze obciążeni żadną opłatą.
# Jest to również istotna informacja, dlatego brak zamieniany jest na 0.
dane["TotalCharges"] = dane["TotalCharges"].fillna(0)

braki = dane.isna().sum()
print(braki[braki > 0])
# Po tej operacji w zbiorze nie ma braków danych.

dane.info()

# Weryfikacja wartości odstających
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

X = dane.drop(columns="Churn")
y = dane["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train = X_train.copy()
X_test = X_test.copy()

print("\nZbiór uczący:", X_train.shape)
print("Zbiór testowy:", X_test.shape)
print("\nUdział klas w całym zbiorze:")
print(y.value_counts(normalize=True).sort_index())
print("\nUdział klas w zbiorze uczącym:")
print(y_train.value_counts(normalize=True).sort_index())
print("\nUdział klas w zbiorze testowym:")
print(y_test.value_counts(normalize=True).sort_index())

# Zbiór uczący zawiera 5634 obserwacje (80%), a testowy 1409 (20%), oba mają 25 zmiennych objaśniających.
# Dzięki stratyfikacji udział klientów, którzy odeszli, wynosi ok. 26,5% we wszystkich zbiorach, więc zbiór testowy jest reprezentatywny. 
# Klasy są niezrównoważone (odchodzi co czwarty klient), dlatego przy ocenie modeli oprócz dokładności trzeba uwzględnić czułość.

# Walidacja krzyżowa
wk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Budowa modeli
# Standaryzacja jedynie w regresji logistycznej i SVM, dopasowana jedynie do zbioru uczącego
# zapis class_weight = "balanced" zapewnia większą wagę dla klasy mniejszościowej

# Regresja logistyczna
reg_log = Pipeline([
    ("scaler", StandardScaler()),("model", LogisticRegression(class_weight="balanced", max_iter=10000)),
])
siatka_reg_log = GridSearchCV(reg_log, {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=wk, scoring="roc_auc", n_jobs=-1)
siatka_reg_log.fit(X_train, y_train)
print("Regresja logistyczna: ", siatka_reg_log.best_params_, siatka_reg_log.best_score_)

# Drzewo decyzyjne
drzewo_dec = DecisionTreeClassifier(class_weight="balanced", random_state=42)
siatka_drzewo_dec = GridSearchCV(drzewo_dec, {"max_depth": [4, 6, 8, 10, 12, 15], "min_samples_leaf": [10, 25, 50, 100]},
                                 cv=wk, scoring="roc_auc", n_jobs=-1)
siatka_drzewo_dec.fit(X_train, y_train)
print("Drzewo decyzyjne: ", siatka_drzewo_dec.best_params_, siatka_drzewo_dec.best_score_)

# Las losowy
las_losowy = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
siatka_las_losowy = GridSearchCV(las_losowy, {"max_depth": [6, 10, 15, None], "min_samples_leaf": [5, 25, 50]},
                                 cv=wk, scoring="roc_auc", n_jobs=-1)
siatka_las_losowy.fit(X_train, y_train)
print("Las losowy: ", siatka_las_losowy.best_params_, siatka_las_losowy.best_score_)

# SVM
svm = Pipeline([("scaler", StandardScaler()), ("model", SVC(class_weight="balanced")),])
siatka_svm = GridSearchCV(svm, [{"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1]},
                                {"model__kernel": ["rbf"], "model__C": [0.01, 0.1, 1],
                                 "model__gamma": [0.0001, 0.001, 0.1, "scale"]}],
                          cv=wk, scoring="roc_auc", n_jobs=-1)
siatka_svm.fit(X_train, y_train)
print("SVM: ", siatka_svm.best_params_, siatka_svm.best_score_)


# Analiza uzyskanych wyników - ocena jakoścu klasyfikacji

# Funkcje pomocnicze
MIARY = ["dokladnosc", "czulosc", "specyficznosc", "AUC"]
ETYKIETY = ["Dokładność", "Czułość", "Specyficzność", "AUC"]

def oblicz_miary(model, x_, y_):
    y_ocena = model.predict_proba(x_)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_)
    tn, fp, fn, tp = confusion_matrix(y_, model.predict(x_), labels=[0,1]).ravel()
    return {
        "dokladnosc": (tp + tn) / (tn + fp + fn + tp),
        "czulosc": tp / (tp + fn),
        "specyficznosc": tn / (tn + fp),
        "AUC": roc_auc_score(y_, y_ocena)
    }

def macierze_pomylek(model, tytul):
    fig, osie = plt.subplots(1, 2, figsize=(9, 3.5))
    for os_, (nazwa, X_, y_) in zip(osie, [("uczący", X_train, y_train), ("testowy", X_test, y_test)]):
        sns.heatmap(confusion_matrix(y_, model.predict(X_)), annot=True, fmt="d", cmap="Blues", ax=os_,
            xticklabels=["został", "odszedł"], yticklabels=["został", "odszedł"])
        os_.set_title(f"Zbiór {nazwa}")
        os_.set_xlabel("Przewidywana klasa")
        os_.set_ylabel("Rzeczywista klasa")
    fig.suptitle(tytul)
    plt.tight_layout()
    plt.show()

# Modele z najlepszymi hiperparametrami
modele = {
    "Regresja logistyczna": siatka_reg_log.best_estimator_,
    "Drzewo decyzyjne": siatka_drzewo_dec.best_estimator_,
    "Las losowy": siatka_las_losowy.best_estimator_,
    "SVM": siatka_svm.best_estimator_
}

miary_cv = {"dokladnosc": "accuracy", "czulosc": "recall", "specyficznosc": make_scorer(recall_score, pos_label=0), "AUC": "roc_auc"}

# Dla każdego modelu: miary na zbiorze uczącym i testowym, macierze pomyłek i 5-krotna walidacja krzyżowa
wiersze_test, wiersze_cv = {}, {}
for nazwa, model in modele.items():
    wyniki = pd.DataFrame({"uczący": oblicz_miary(model, X_train, y_train),
                           "testowy": oblicz_miary(model, X_test, y_test)}).T
    print(f"\n{nazwa}")
    print(wyniki.round(3))
    macierze_pomylek(model, nazwa)

    wiersze_test[nazwa] = wyniki.loc["testowy"]
    cv = cross_validate(model, X_train, y_train, cv=wk, scoring=miary_cv, n_jobs=-1)
    wiersze_cv[nazwa] = {m: cv[f"test_{m}"].mean() for m in MIARY}

# W zbiorze testowym jest 374 klientów, którzy odeszli i 1045, którzy zostali

# Regresja logistyczna: na zbiorze testowym dokładność 0,74, czułość 0,775, specyficznośc 0,728 i AUC 0,837
# Model nie wykrywa 84 z 374 odejść (22,5%), a 282 z 1035 klientów, którzy zostali (27,2%), błędnie wskazuje jako zagrożonych odejściem
# Wyniki na zbiorze uczącym są tylko nieznacznie wyższe, więc model nie jest przeuczony

# Drzewo decyzyjne: dokładność 0,751, czułość 0,797, specyficzność 0,734 i AUC 0,835
# Pomija 76 odejść (20,3%) i daje 275 fałszywych alarmów (26,6%)
# Różnica między zbiorem uczącym i testowym jest największa w czułości (0,831 vs 0,797) i AUC (0,865 vs 0,835), ale nadal niewielka

# Las losowy: dokładność 0,760, czułość 0,797, specyficzność 0,747 i AUC 0,843
# Pomija 76 odejść (20,3%) i daje najmniej fałszywych alarmow (262, czyli 25,3%)
# Różnica AUC między zbiorem uczącym i testowym wynosi 0,029, więc przeuczenie jest niewielkie

# SVM: dokładność 0,702, czułość 0,821, specyficzność 0,659 i AUC 0,835
# Pomija najmniej odejść (67, czyli 17,9%), ale daje najwięcej fałszywych alarmów (353, czyli 34,1%)
# Wyniki na obu zbiorach są niemal identyczne
# Najlepsze jądro to rbf z małym gamma (0,001), czyli granica decyzyjna bliska liniowej

# We wszystkich modelach czułość jest wyższa niż specyficzność


# Porównanie modeli
porownanie_test = pd.DataFrame(wiersze_test).T
porownanie_cv = pd.DataFrame(wiersze_cv).T
for df in (porownanie_test, porownanie_cv):
    df["balanced_accuracy"] = (df["czulosc"] + df["specyficznosc"]) / 2

print("Zbiór testowy:")
print(porownanie_test.round(3).to_string())
print("\nŚrednia z 5-krotnej walidacji krzyżowej (zbiór uczący):")
print(porownanie_cv.round(3).to_string())

# Wszystkie modele mają AUC ok. 0,83–0,85, zarówno na zbiorze testowym, jak i w walidacji krzyżowej, więc dobrze odróżniają klientów, którzy odejdą, od tych, którzy zostaną
# Dokładność (0,70–0,75) jest zbliżona do dokładności modelu, który zawsze przewiduje „został” (0,735), ale taki model miałby czułość równą 0, a modele wykrywają ok. 78–82% odejść
# Pokazuje to, że przy niezbalansowanych danych sama dokładność nie wystarcza do oceny modelu

# Wyniki na zbiorze testowym i w walidacji krzyżowej są spójne
# Las losowy ma najwyższe AUC w obu przypadkach (0,843 i 0,848)
# Regresja logistyczna w walidacji krzyżowej wypada niewiele słabiej niż las losowy (AUC 0,845 vs 0,848, dokładność 0,748 vs 0,753)
# Drzewo decyzyjne nieco słabiej (AUC 0,833, specyficzność 0,712)
# SVM ma najwyższą czułość (0,835 w walidacji krzyżowej), ale wyraźnie najniższą specyficzność (0,659) i dokładność (0,706)


fig, osie = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
for os_, (tytul, df) in zip(osie, [("Zbiór testowy", porownanie_test), ("Walidacja krzyżowa (5 części)", porownanie_cv)]):
    df[MIARY].set_axis(ETYKIETY, axis=1).plot(kind="bar", ax=os_, rot=0)
    os_.set_title(tytul)
    os_.set_ylim(0, 1.15)
    os_.legend(loc="upper center", ncol=4)
    os_.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Krzywe ROC na zbiorze testowym
fig, ax = plt.subplots(figsize=(7, 6))
for nazwa, model in modele.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=nazwa)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Klasyfikator losowy")
ax.set_title("Krzywe ROC – zbiór testowy")
ax.legend()
plt.show()

# Krzywe ROC pokazują czułość i odsetek fałszywych alarmów (1 - specyficzność) przy wszystkich możliwych progach odcięcia
# Krzywe wszystkich modeli niemal się pokrywają i leżą wyraźnie powyżej przekątnej
# Potwierdza to że modele mają bardzo podobną zdolność rozróżniania klas, a różnią się głównie położeniem progu odcięcia

# Podsumowanie porównania modeli:
# Las losowy ma najlepsze wyniki: najwyższe AUC i balanced accuracy zarówno na zbiorze testowym, jak i w walidacji krzyżowej, przy czułosci ok. 0,80 i specyficzności ok. 0,74-0,75.
# Regresja logistyczna osiąga prawie takie same wyniki w walidacji krzyżowej
# Drzewo decyzyjne wypada dobrze na zbiorze testowym, ale w walidacji krzyżowej nieco słabiej niż las losowy i regresja logistyczna
# SVM wykrywa najwięcej odejść, ake kosztem największej liczby fałszywych alarmów

# Różnice między modelami są niewielkie, więc wybór zależy od celu
# Jako model końcowy można wybrac las losowy
# Gdyby priorytetem było wykrycie jak największej liczby odchodzących klientów, można rozwazyć SVM
