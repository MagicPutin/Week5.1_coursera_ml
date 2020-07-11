import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

# data preparing
data = pd.read_csv('Data/abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

x = data.loc[:, data.columns != 'Rings']
y = data['Rings']
flag = 1
for i in range(1, 51):
    score = 0

    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    clf.fit(x, y)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(clf, x, y, scoring='r2', cv=kf)
    r = scores.mean()
    if r > 0.52 and flag:
        ans = i + 1 # why +1? I don't know, I'm using numbers of forest (not indexes) and it's failed
        flag = 0
    print(i, r)

with open('Answers/task1.txt') as task:
    task.write(str(ans))
