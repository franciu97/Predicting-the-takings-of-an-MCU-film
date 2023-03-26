#Librerie e funzioni importate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

#Caricamento e analisi iniziale dei dati
data = pd.read_csv('movie_info.csv')
print(data.head())

print(data.info())
print(data.describe())

# Calcolo della matrice di correlazione
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Seleziona solo le correlazioni relative alla variabile target
target_corr = corr_matrix['worldwide_collection_in_million_(USD)']

# Stabilisce una soglia di correlazione (ad esempio, 0.1)
threshold = 0.1

# Seleziona le colonne con una correlazione superiore alla soglia
selected_columns = target_corr[abs(target_corr) > threshold].index

# Mantiene solo le colonne selezionate nel dataset
data = data[selected_columns]

# Preparazione del dataset
X = data.drop('worldwide_collection_in_million_(USD)', axis=1)
y = data['worldwide_collection_in_million_(USD)']

# Divisione in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Modelli da confrontare
models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=5),
    'Random Forest': RandomForestRegressor(random_state=5),
}

# Addestramento e valutazione dei modelli
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'{name} RMSE: {rmse}')

# Cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    print(f'{name} CV RMSE: {-scores.mean()}')

# Definizione della griglia di parametri per l'ottimizzazione degli iperparametri
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Ottimizzazione degli iperparametri con ricerca esaustiva Grid Search
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X, y)
print('Best parameters:', grid_search.best_params_)

# Addestramento e valutazione del modello ottimizzato
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{name} RMSE: {rmse}')
    print(f'{name} MSE: {mse}')
    print(f'{name} R²: {r2}')
    print()

# Distribuzione degli incassi
sns.histplot(data['worldwide_collection_in_million_(USD)'], kde=True)
plt.xlabel('Box Office')
plt.ylabel('Conteggio')
plt.title('Distribuzione degli incassi')
plt.show()

# Scatterplot tra budget e incasso
sns.scatterplot(x=data['production_budget_in_million_(USD)'], y=data['worldwide_collection_in_million_(USD)'])
plt.xlabel('Budget')
plt.ylabel('Box Office')
plt.title('Budget vs Box Office')
plt.show()

# Scatterplot tra valutazione degli spettatori e incasso
sns.scatterplot(x=data['meta_user_score'], y=data['worldwide_collection_in_million_(USD)'])
plt.xlabel('Punteggio del pubblico')
plt.ylabel('Box Office')
plt.title('Punteggio del pubblico vs Box Office')
plt.show()

# Scatterplot tra valutazione della critica e incasso
sns.scatterplot(x=data['metascore'], y=data['worldwide_collection_in_million_(USD)'])
plt.xlabel('Punteggio della critica')
plt.ylabel('Box Office')
plt.title('Punteggio della critica vs Box Office')
plt.show()





