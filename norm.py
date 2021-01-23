import pandas as pd
import ipaddress
from sklearn import preprocessing

data = pd.read_csv('csvallpack.csv')
nrow = int(data['Time'].count())

for i in range(nrow):
    data['Source'][i] = int (ipaddress.ip_address(data['Source'][i]))
    data['Destination'][i] = int (ipaddress.ip_address(data['Destination'][i]))



data['Source']=(data['Source']-data['Source'].min())/(data['Source'].max()-data['Source'].min())
data['Destination']=(data['Destination']-data['Destination'].min())/(data['Destination'].max()-data['Destination'].min())
data['Time']=(data['Time']-data['Time'].min())/(data['Time'].max()-data['Time'].min())
data['Length']=(data['Length']-data['Length'].min())/(data['Length'].max()-data['Length'].min())

df = data[['Time','Source','Destination','Protocol','Length']]

df_dummies = pd.get_dummies(data['Protocol'])
pd.concat([df, df_dummies], axis=1)

df.to_csv('normout.csv',index=False)
