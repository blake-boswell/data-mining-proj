import pandas as pd
import numpy as np

# Read in the sessions.csv file
df = pd.read_csv('./airbnb-recruiting-new-user-bookings/sessions.csv', header=0)

# Grab all instances of a user on a device, and add them up into one duration
print(df.columns)
# :, [] appears to give for columns
userDevices = df.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
print(userDevices)

# Merge same device into one record by summing secs elapsed
print(userDevices.groupby(['user_id','device_type'])['secs_elapsed'].sum())