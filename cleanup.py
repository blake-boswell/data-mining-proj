import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~ Train and Test clean up ~~~~~~~~~~~~~~~~~

def oneHotEncode(df, column):
    # Get field values
    categories = list(df[column].drop_duplicates())
    for category in categories:
        categoryName = str(category).replace(" ", "-").replace("/", "-").lower()
        columnName = column + '_' + categoryName
        df[columnName] = 0
        df.loc[(df[column] == category), columnName] = 1
    return df

print('Starting to clean test and train data')
# Read in the train and test data
dfTest = pd.read_csv('./airbnb-recruiting-new-user-bookings/test_users.csv', header=0)
dfTrain = pd.read_csv('./airbnb-recruiting-new-user-bookings/train_users_2.csv', header=0)

# Fix ages > 100 and less than 15 (not realistic)
dfTest['age'] = np.where(np.logical_or((dfTest['age'].values < 12), (dfTest['age'].values > 100)), np.NaN, dfTest['age'].values)
dfTrain['age'] = np.where(np.logical_or((dfTrain['age'].values < 12), (dfTrain['age'].values > 100)), np.NaN, dfTrain['age'].values)

# Fix timestamps (set to datetime values for date functions) while keeping the data
dfTest['date_account_created'] = pd.to_datetime(dfTest['date_account_created'], format='%Y-%m-%d')
dfTest['timestamp_first_active'] = pd.to_datetime(dfTest['timestamp_first_active'], format='%Y%m%d%H%M%S')

dfTrain['date_account_created'] = pd.to_datetime(dfTrain['date_account_created'], format='%Y-%m-%d')
dfTrain['timestamp_first_active'] = pd.to_datetime(dfTrain['timestamp_first_active'], format='%Y%m%d%H%M%S')

# Remove date_first_booking field. It's unused in the test_users, so it can't give any insight
dfTest.drop('date_first_booking', axis='columns', inplace=True)
dfTrain.drop('date_first_booking', axis='columns', inplace=True)

# Set all Na values to -1 (age, date_account_created, first_affiliate_tracked)
dfTest['age'].fillna(-1, inplace=True)
dfTrain['age'].fillna(-1, inplace=True)

# Fill in empty date_account_created values by using the timestamp
dfTest['date_account_created'].fillna(dfTest['timestamp_first_active'], inplace=True)
dfTest['first_affiliate_tracked'].fillna(-1, inplace=True)

dfTrain['date_account_created'].fillna(dfTrain['timestamp_first_active'], inplace=True)
dfTrain['first_affiliate_tracked'].fillna(-1, inplace=True)

# Add new data from the date columns (date_account_created, timestamp_first_active)
dfTest['hour_first_active'] = dfTest['timestamp_first_active'].dt.hour
dfTest['day_first_active'] = dfTest['timestamp_first_active'].dt.weekday
dfTest['month_first_active'] = dfTest['timestamp_first_active'].dt.month
dfTest['quarter_first_active'] = dfTest['timestamp_first_active'].dt.quarter
dfTest['year_first_active'] = dfTest['timestamp_first_active'].dt.year
dfTest['day_account_created'] = dfTest['date_account_created'].dt.weekday
dfTest['month_account_created'] = dfTest['date_account_created'].dt.month
dfTest['quarter_account_created'] = dfTest['date_account_created'].dt.quarter
dfTest['year_account_created'] = dfTest['date_account_created'].dt.year

dfTrain['hour_first_active'] = dfTrain['timestamp_first_active'].dt.hour
dfTrain['day_first_active'] = dfTrain['timestamp_first_active'].dt.weekday
dfTrain['month_first_active'] = dfTrain['timestamp_first_active'].dt.month
dfTrain['quarter_first_active'] = dfTrain['timestamp_first_active'].dt.quarter
dfTrain['year_first_active'] = dfTrain['timestamp_first_active'].dt.year
dfTrain['day_account_created'] = dfTrain['date_account_created'].dt.weekday
dfTrain['month_account_created'] = dfTrain['date_account_created'].dt.month
dfTrain['quarter_account_created'] = dfTrain['date_account_created'].dt.quarter
dfTrain['year_account_created'] = dfTrain['date_account_created'].dt.year

# One-hot-encode option
columns = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for column in columns:
    dfTest = oneHotEncode(dfTest, column)
    dfTest.drop(column, axis='columns', inplace=True)

    dfTrain = oneHotEncode(dfTrain, column)
    dfTrain.drop(column, axis='columns', inplace=True)

# Write to modified csv file
print('Writing modified train and test data to files...')
dfTrain.set_index('id', inplace=True)
dfTest.set_index('id', inplace=True)
dfTrain.to_csv('./output/modified_train_users.csv', index_label='id')
dfTest.to_csv('./output/modified_test_users.csv', index_label='id')

# ~~~~~~~~~~~~~~ Sessions clean up ~~~~~~~~~~~~~~~

print('Starting session clean up')
# Read in the sessions.csv file
dfSession = pd.read_csv('./airbnb-recruiting-new-user-bookings/sessions.csv', header=0)

# Grab all instances of a user on a device, and add them up into one duration
# print(dfSession.columns)
# :, [] appears to give for columns
userDevices = dfSession.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
# print(userDevices)

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
    # print('----------------------')
    # print(newDf.loc[df[column] == '10'])
    # print('----------------------')
    # print(newDf)
    # print(newDf[column].values)
    newDf = newDf.pivot(index='user_id', columns=column, values='count')
    newDf.fillna(0, inplace=True)
    # print(newDf)
    # 3. Rename the columns created
    columns = list(df[column].drop_duplicates())
    # print(columns)
    for col in columns:
        colName = str(col).replace(" ", "-").replace("/", "-").lower()
        newCategory = column + '_' + str(colName)
        newDf.rename(columns={ col: newCategory }, inplace=True)
    return newDf

# Get the counts of each action
# 1. Loop over each action col
actionColumns = ['action', 'action_type', 'action_detail']
actionDf = dfSession.loc[:, ['user_id', 'action', 'action_type', 'action_detail']]
actionDf = actionDf.fillna('NA')
actionDf['user_id'].drop_duplicates()
transformedActionDf = actionDf
print('Transforming action columns to hold counts')
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
combinedSessionDf = combinedSessionDf.fillna(0)

# combinedSessionDf.to_csv('./output/modified_session.csv')

# Merge sessions and train files by the user id
dfSessionTrain = pd.concat([dfTrain, combinedSessionDf], join='outer', axis=1)
print('Writing the session and train joined data file.')
dfSessionTrain.to_csv('./output/modified_session_train.csv', index_label='id')
print('Complete. Use ./output/modified_session_train.csv for the training data, and ./output/modified_test_users.csv as the test data')


