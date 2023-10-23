# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos (reemplaza 'train.csv' con la ubicación de tu archivo CSV)
#d = pd.read_csv("G:\disco portatil 5 gigas\Mis Documentos\Visual Studio 2022\proyectos python\proyecto2\train.csv")
#d = pd.read_csv("G:/disco portatil 5 gigas/Mis Documentos/Visual Studio 2022/proyectos python/proyecto2/train.csv")
d = pd.read_csv("G:\\disco portatil 5 gigas\\Mis Documentos\\Visual Studio 2022\\proyectos python\\proyecto2\\train.csv")


# Comprueba las primeras filas de los datos
print(d.head())

# Ver el tamaño de los datos
print(d.shape)

# Comprobar valores faltantes en las columnas
k = d.isna().sum()
print(k[k != 0])

# Visualizar la variable objetivo
#sns.distplot(d['SalePrice'])
sns.distplot(d['P_2'])
P_2
# Descubre los tipos de datos de las columnas
print(d.dtypes)

# Inspeccionar columnas numéricas
numeric_cols = d.select_dtypes(include=[np.number])
print(numeric_cols.describe())

# Visualizar relaciones entre algunas variables numéricas
#cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
cols = ['D_2', 'S_2', 'B_2', 'R_2', 'P_2']
sns.pairplot(d[cols])

# Visualizar correlaciones entre variables numéricas
corrmat = d.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)

# Inspeccionar variables categóricas
categorical_cols = d.select_dtypes(exclude=[np.number])
print(categorical_cols.columns)

for c in categorical_cols:
    print(c)
    print(d[c].unique())

# Ejemplo de visualización de variables categóricas
c = "GarageType"
print(d[c].value_counts())

# Visualización de variables categóricas
plt.figure(figsize=(20, 8))
for i, c in enumerate(["ExterQual", "HouseStyle", "LandSlope", "Alley"]):
    plt.subplot(2, 4, i + 1)
    #k = d[[c, "SalePrice"]].dropna()
    k = d[[c, "P_2"]].dropna()
    for v in d[c].dropna().unique():
        #sns.distplot(k.SalePrice[k[c] == v], label=v)
        sns.distplot(k.P_2[k[c] == v], label=v)
        plt.title(c)
    plt.yticks([])
    plt.legend()
    plt.subplot(2, 4, i + 5)
    vc = k[c].value_counts()
    sns.barplot(vc.index, vc.values)
    plt.xticks(range(len(vc)), vc.index, rotation="vertical")

# Visualizar valores faltantes en las columnas
k = d.isna().sum()
print(k[k != 0])
