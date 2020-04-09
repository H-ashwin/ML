import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv("music.csv")
print(music_data)
# split the data input set and output set
x = music_data.drop(columns=['genre'])   # first splitting for training
# print(x)
y = music_data['genre']    # and second for Testing
# print(y)

model = DecisionTreeClassifier()
model.fit(x, y)  # here we are training the model
predictions = model.predict([[21, 1], [22, 0]])
print(predictions)
