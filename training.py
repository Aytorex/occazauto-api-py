import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

global df
df = pd.read_csv('C:/Users/Joshua/PycharmProjects/occazauto-api-v3/data/carsTDIA.csv', sep=';')


def convertmanufacturer_name(manufacturer: str, columndataframe):
    index = df['manufacturer_name'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def convertmodel_name(manufacturer: str, columndataframe):
    index = df['model_name'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def converttransmission(manufacturer: str, columndataframe):
    index = df['transmission'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def convertcolor(manufacturer: str, columndataframe):
    index = df['color'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def convertengine_fuel(manufacturer: str, columndataframe):
    index = df['engine_fuel'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def convertengine_type(manufacturer: str, columndataframe):
    index = df['engine_type'].loc[lambda x: x == manufacturer].index[0]
    return columndataframe[index]


def predictannouce(manufacturer_name: str, model_name: str, transmision: str, color: str, odometer_value: int,
                   year: int, engine_fuel: str, idengine_type: str, price: float):
    le = LabelEncoder()
    X0 = df.iloc[:, 0]  # extraction des colonnes une par une
    X1 = df.iloc[:, 1]
    X2 = df.iloc[:, 2]
    X3 = df.iloc[:, 3]
    X4 = df.iloc[:, 4]
    X5 = df.iloc[:, 5]
    X6 = df.iloc[:, 6]
    X7 = df.iloc[:, 7]
    X8 = df.iloc[:, 8]

    X0 = le.fit_transform(X0)  # conversion en int pour les données en string
    X1 = le.fit_transform(X1)
    X2 = le.fit_transform(X2)
    X3 = le.fit_transform(X3)
    X6 = le.fit_transform(X6)
    X7 = le.fit_transform(X7)

    try:
        idmanufacturer_name = convertmanufacturer_name(manufacturer_name, X0)  # conversion no
    except:
        return "Erreur: Le constructeur est mal orthographié ou est inconnu de notre agence"
    try:
        idmodel_name = convertmodel_name(model_name, X1)
    except:
        return "Erreur: Le modèle est mal orthographié ou est inconnu de notre agence"
    try:
        idtransmision = converttransmission(transmision, X2)
    except:
        return "Erreur: La transmission est mal orthographié ou est inconnu de notre agence"
    try:
        idcolor = convertcolor(color, X3)
    except:
        return "Erreur: Cette couleur est mal orthographié ou est inconnu de notre agence, priviligiez des nom de couleur strandard, ex: bleu et pas bleu fusion"
    try:
        idengine_fuel = convertengine_fuel(engine_fuel, X6)
    except:
        return "Erreur: Ce type de carburant est mal orthographié ou est inconnu de notre agence"
    try:
        idengine_type = convertengine_type(idengine_type, X7)
    except:
        return "Erreur: Ce type de moteur est mal orthographié ou est inconnu de notre agence"

    X = np.column_stack((X0, X1, X2, X4, X5, X3, X6, X7))  # concaténation de toutes les colonnes
    Y = X8  # sauf le prix qui sert à trust l'annonce

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)  # 80% des données pour l'entrainement, le reste pour les tests

    rf = pickle.load(open("C:/Users/Joshua/PycharmProjects/occazauto-api-v3/extra-tree-regressor.sav", 'rb'))

    Y_rf_train_pred = rf.predict(X_train)  # tentative de prédiction avec le modèle entrainé
    Y_rf_test_pred = rf.predict(X_test)  # pareil avec le test

    testdata = [idmanufacturer_name, idmodel_name, idtransmision, odometer_value, year, idcolor, idengine_fuel,
                idengine_type]  # données envoyé par l'utilisateur (déjà converties en int)

    test = rf.predict(np.array(testdata).reshape((1, -1)))  # n'accépte pas les tableau à une dimension, reformatage

    maxval = max(test, price)
    minval = min(test, price)
    return float((1 - (maxval - minval) / (maxval)))  # calcul proba par écart à la moyenne


print(float(predictannouce("Subaru", "Outback", "automatic", "silver", 190000, 2010, "gasoline", "gasoline", 10900.0)))
