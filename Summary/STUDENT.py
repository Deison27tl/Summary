#DEISON TUIRAN LONDOÑO
#ID 014810
#ID 1003644616


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scale = StandardScaler()


student = pd.read_csv("student_data.csv")
print(student.corr())
x = student[["G1","G2"]]
y = student[["G3"]]

scaledX = scale.fit_transform(x)



train_x = scaledX[:316]
train_y = y[:316]

test_x = scaledX[316:]
test_y = y[316:]


model = linear_model.LinearRegression().fit(train_x, train_y)

print(model.score(train_x,train_y))

print(model.score(test_x, test_y))
