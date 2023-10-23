# Importa las bibliotecas necesarias
from pyspark.sql import SparkSession
import seaborn as sns
import matplotlib.pyplot as plt

# Crea una instancia de SparkSession
spark = SparkSession.builder.appName("DataExploration").getOrCreate()

# Carga el archivo CSV con Spark
df = spark.read.csv("G:\\disco portatil 5 gigas\\Mis Documentos\\Visual Studio 2022\\proyectos python\\proyecto2\\train.csv", header=True, inferSchema=True)


# Comprueba las primeras filas de los datos
df.show()

# Ver el tamaño de los datos
print((df.count(), len(df.columns)))

# Comprobar valores faltantes en las columnas
missing_cols = [col for col in df.columns if df.filter(df[col].isNull()).count() > 0]
for col in missing_cols:
    print(col, df.filter(df[col].isNull()).count())

# Visualizar la variable objetivo
#sns.distplot(df.select("SalePrice").rdd.flatMap(lambda x: x).collect())
sns.distplot(df.select("P_2").rdd.flatMap(lambda x: x).collect())
# Descubre los tipos de datos de las columnas
for col in df.columns:
    print(col, df.schema[col].dataType)

# Inspeccionar columnas numéricas
numeric_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]
df.select(numeric_cols).describe().show()

# Visualizar relaciones entre algunas variables numéricas
#cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
cols = ['D_2', 'S_2', 'B_2', 'R_2', 'P_2']
numeric_df = df.select(cols)
sns.pairplot(numeric_df.toPandas())

# Visualizar correlaciones entre variables numéricas
numeric_data = df.select(cols)
numeric_data = numeric_data.toPandas()
corrmat = numeric_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)

# Inspeccionar variables categóricas
categorical_cols = [col for col, dtype in df.dtypes if dtype == 'string']
print(categorical_cols)

for col in categorical_cols:
    print(col)
    unique_values = df.select(col).distinct().rdd.flatMap(lambda x: x).collect()
    print(unique_values)

# Ejemplo de visualización de variables categóricas
c = "GarageType"
df.groupBy(c).count().show()

# Visualización de variables categóricas
plt.figure(figsize=(20, 8))
for i, c in enumerate(["ExterQual", "HouseStyle", "LandSlope", "Alley"]):
    plt.subplot(2, 4, i + 1)
    #k = df.select(c, "SalePrice").na.drop().toPandas()
    k = df.select(c, "P_2").na.drop().toPandas()
    #for v in k.select("SalePrice").rdd.flatMap(lambda x: x).collect():
    for v in k.select("P_2").rdd.flatMap(lambda x: x).collect():    
        #sns.distplot(k[k[c] == v].select("SalePrice").rdd.flatMap(lambda x: x).collect(), label=v)
        sns.distplot(k[k[c] == v].select("P_2").rdd.flatMap(lambda x: x).collect(), label=v)
        plt.title(c)
    plt.yticks([])
    plt.legend()
    plt.subplot(2, 4, i + 5)
    vc = k.groupBy(c).count().toPandas()
    sns.barplot(vc[c], vc["count"])
    plt.xticks(range(len(vc)), vc[c], rotation="vertical")

# Visualizar valores faltantes en las columnas
missing_cols = [col for col in df.columns if df.filter(df[col].isNull()).count() > 0]
for col in missing_cols:
    print(col, df.filter(df[col].isNull()).count())
