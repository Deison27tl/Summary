#Taller 
#Summary
#Gariela Torres
#ID:1001970935
#ID:502193
#correo:gabriela.torresr@correo.upb.edu.co
#Cel:3234708201
#Diplomado de PYTHON APLICADO A LA INGENIERIA 
#Docente:Roberto Paez Salgado
#Modulo 2

#Importar librerias
import pandas as pd
#DEISON TUIRAN LONDOÑO
#ID 014810
#ID 1003644616


from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


scale = StandardScaler()


df = pd.read_csv("cars2.csv")
x = df[["Volume","Weight"]]
y = df[["CO2"]]


scaledX = scale.fit_transform(x)


train_x = scaledX[:28]
train_y = y[:28]

test_x = scaledX[28:]
test_y = y[28:]


model = linear_model.LinearRegression().fit(train_x, train_y)

print(model.score(train_x,train_y))

print(model.score(test_x, test_y))


