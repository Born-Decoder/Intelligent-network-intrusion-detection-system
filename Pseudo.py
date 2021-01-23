import random
import pandas as pd

data = pd.read_csv('normout.csv')
nrow = int(data['Time'].count())

res = [random.randint(0,1)for i in range(nrow)]

data['Safe'] = res

data.to_csv('normout.csv',index=False)

