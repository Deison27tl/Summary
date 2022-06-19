#DEISON TUIRAN LONDOÑO
#ID 014810
#ID 1003644616


from email.policy import default
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scale = StandardScaler()


df_netflix   = pd.read_excel("df_netflix _list.xlsx").dropna()


condiciones  = [
    (df_netflix  ["type"] == "Movie"),
    (df_netflix  ["type"] == "TV Show")
]

eleccion_lista = [0,1]
df_netflix  ["type_normalized"] = np.select(condiciones , eleccion_lista, default = "Not_specified")
separate_duration_movies = df_netflix  ["duration"].str.split(expand=True)

df_netflix  .insert(4,"durationInt", separate_duration_movies[0].astype(int))
print(df_netflix  )

x,y = df_netflix  ["durationInt"], df_netflix  ["type_normalized"]
x,y = np.array(x).reshape(-1,1), np.array(y)
scaledX = scale.fit_transform(x)
train_X, train_y = scaledX[:1000], y[:1000]
test_x, test_y = scaledX[1000:1200], y[1000:1200]

plt.scatter(scaledX[:100], y[:100])
plt.show()

model = linear_model.LinearRegression().fit(train_X, train_y)

print(model.score(train_X, train_y))
print(model.score(test_x, test_y))
