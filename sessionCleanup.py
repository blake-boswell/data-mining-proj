import pandas as pd
import numpy as np

def oneHotEncode(df, column):
    # Get field values
    categories = list(df[column].drop_duplicates())
    for category in categories:
        categoryName = str(category).replace(" ", "-").replace("/", "-").lower()
        columnName = column + '_' + categoryName
        df[columnName] = 0
        df.loc[(df[column] == category), columnName] = 1
    return df

# Read in the sessions.csv file
df = pd.read_csv('./airbnb-recruiting-new-user-bookings/sessions.csv', header=0)

# Grab all instances of a user on a device, and add them up into one duration
print(df.columns)
# :, [] appears to give for columns
userDevices = df.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
print(userDevices)

# Merge same device into one record by summing secs elapsed
userDevices = userDevices.groupby(['user_id','device_type'], sort=False, as_index=False)['secs_elapsed'].aggregate(np.sum)
# Get the index of the primary device used for the user
index = userDevices.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == userDevices['secs_elapsed']
primaryDevices = pd.DataFrame(userDevices.loc[index, ['user_id', 'device_type', 'secs_elapsed']])
primaryDevices.rename(columns = { 'device_type': 'primary_device', 'secs_elapsed': 'primary_secs_elapsed' }, inplace = True)
primaryDevices.set_index('user_id', inplace=True)

otherDevices = userDevices.drop(userDevices.index[index])
index = otherDevices.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == otherDevices['secs_elapsed']
secondaryDevices= pd.DataFrame(otherDevices.loc[index, ['user_id', 'device_type', 'secs_elapsed']])
secondaryDevices.rename(columns = { 'device_type': 'secondary_device', 'secs_elapsed': 'secondary_secs_elapsed' }, inplace = True)
secondaryDevices.set_index('user_id', inplace=True)

deviceDf = pd.concat([primaryDevices, secondaryDevices], join='outer', axis=1)

# primaryDevices = oneHotEncode(primaryDevices, 'primary_device')
# primaryDevices.drop(columns = 'primary_device', inplace = True)

# print(primaryDevices)
# print(primaryDevices.columns)

# print(secondaryDevices)
# print(secondaryDevices.columns)

# 2. Get the counts of that action for each user
def countsTransform(df, column):
    newDf = df.loc[:, ['user_id', column]]
    newDf['count'] = 1
    newDf = newDf.groupby(['user_id', column], sort=False, as_index=False).sum()
    print('----------------------')
    print(newDf.loc[df[column] == '10'])
    print('----------------------')
    print(newDf)
    print(newDf[column].values)
    newDf = newDf.pivot(index='user_id', columns=column, values='count')
    newDf.fillna(0, inplace=True)
    print(newDf)
    # 3. Rename the columns created
    columns = list(df[column].drop_duplicates())
    print(columns)
    for col in columns:
        colName = str(col).replace(" ", "-").replace("/", "-").lower()
        newCategory = column + '_' + str(colName)
        newDf.rename(columns={ col: newCategory }, inplace=True)
    return newDf

# Get the counts of each action
# 1. Loop over each action col
actionColumns = ['action', 'action_type', 'action_detail']
actionDf = df.loc[:, ['user_id', 'action', 'action_type', 'action_detail']]
actionDf = actionDf.fillna('NA')
actionDf['user_id'].drop_duplicates()
transformedActionDf = actionDf
print('Somethings going on')
concatenate = False
for actionCol in actionColumns:
    transformedDf = countsTransform(df=actionDf, column=actionCol)
    if concatenate:
        transformedActionDf = pd.concat([transformedActionDf, transformedDf], join='inner', axis=1)
    else:
        transformedActionDf = transformedDf
        concatenate = True
    print(str(actionCol) + ' done.')

# 2. Get the counts of that action for each user
# actions = df.loc[:, ['user_id', 'action']]
# print('----------------------')
# # print(actions.loc[actions['action'] == '10'])
# print('----------------------')
# actionCounts = actions.fillna('NA')
# # print(actions)
# # print()

# Combine the sets
combinedSessionDf = pd.concat([deviceDf, transformedActionDf], join='outer', axis=1)
# combinedSessionDf = combinedSessionDf.fillna()

print(combinedSessionDf)
combinedSessionDf.to_csv('modified_session.csv')