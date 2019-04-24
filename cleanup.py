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

# Read in the train and test data
dfTest = pd.read_csv('./airbnb-recruiting-new-user-bookings/test_users.csv', header=0)
dfTrain = pd.read_csv('./airbnb-recruiting-new-user-bookings/train_users_2.csv', header=0)

# Combine for cleaning
dfCombo = pd.concat((dfTest, dfTrain))

print(dfCombo)

# Fix ages > 100 and less than 15 (not realistic)
print(dfCombo['age'].max())
print(dfCombo.nlargest(25, 'age'))
dfCombo['age'] = np.where(np.logical_or((dfCombo['age'].values < 12), (dfCombo['age'].values > 100)), np.NaN, dfCombo['age'])
print(dfCombo.nlargest(25, 'age'))

# Fix timestamps (set to datetime values for date functions) while keeping the data
print(dfCombo.dtypes)
dfCombo['date_account_created'] = pd.to_datetime(dfCombo['date_account_created'], format='%Y-%m-%d')
dfCombo['timestamp_first_active'] = pd.to_datetime(dfCombo['timestamp_first_active'], format='%Y%m%d%H%M%S')
print()
print(dfCombo.dtypes)

# Remove date_first_booking field. It's unused in the test_users, so it can't give any insight
dfCombo.drop('date_first_booking', axis='columns', inplace=True)
print()
print(dfCombo.dtypes)

# Set all Na values to -1 (age, date_account_created, first_affiliate_tracked)
dfCombo['age'].fillna(-1, inplace=True)
# Fill in empty date_account_created values by using the timestamp
dfCombo['date_account_created'].fillna(dfCombo['timestamp_first_active'], inplace=True)
dfCombo['first_affiliate_tracked'].fillna(-1, inplace=True)

# Add new data from the date columns (date_account_created, timestamp_first_active)
dfCombo['hour_first_active'] = dfCombo['timestamp_first_active'].dt.hour
dfCombo['day_first_active'] = dfCombo['timestamp_first_active'].dt.weekday
dfCombo['month_first_active'] = dfCombo['timestamp_first_active'].dt.month
dfCombo['quarter_first_active'] = dfCombo['timestamp_first_active'].dt.quarter
dfCombo['year_first_active'] = dfCombo['timestamp_first_active'].dt.year
dfCombo['day_account_created'] = dfCombo['date_account_created'].dt.weekday
dfCombo['month_account_created'] = dfCombo['date_account_created'].dt.month
dfCombo['quarter_account_created'] = dfCombo['date_account_created'].dt.quarter
dfCombo['year_account_created'] = dfCombo['date_account_created'].dt.year

# One-hot-encode option
columns = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for column in columns:
    dfCombo = oneHotEncode(dfCombo, column)
    dfCombo.drop(column, axis='columns', inplace=True)
print()
print(dfCombo.columns)



