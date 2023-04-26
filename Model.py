import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

bats = pd.read_excel('/Users/saipranayreddy/Downloads/IPL 2021 (1).xlsx', sheet_name='Sheet1')
bowls = pd.read_excel('/Users/saipranayreddy/Downloads/IPL 2021 (1).xlsx', sheet_name='Sheet2')

#For Batsmen data
bats.isnull().sum() #None
bats.duplicated().sum() #None



#For Bowler Data
bowls.isnull().sum() #None
bowls.duplicated().sum() #None


'''Exploratory Data Analysis for Batsmen data'''
bats.dtypes #Checking for inconsistent data types and found none

print("Shape of the dataset for batsmen:", bats.shape)
#Shape of the dataset for batsmen: (81, 38)

#Summary statistics for numeric columns and also for strings
num_bat = bats.describe()
strn_bowl = bats.describe(include='object')

#Plotting the distribution of the target variable
sns.histplot(data=bats, x='Players_Total_Auction_price')
plt.title('Distribution of Total Auction Price for batsmen')
plt.show()

#Correlation matrix

corr_matrix = bats.corr()
# Filter out correlations that are not greater than 0.7 or less than -0.7
corr_matrix = corr_matrix[(corr_matrix > 0.7) | (corr_matrix < -0.7)]
plt.figure(figsize=(16, 8))
sns.heatmap(corr_matrix, annot=True)


mapping = {'Indian': 1, 'Overseas': 0}
bats['Origin'] = bats['Origin'].map(mapping)
bats['Origin'].fillna(1, inplace=True)
bats1 = bats.drop(['Player','Team in IPL'], axis=1)
bats1 = pd.get_dummies(bats1, drop_first= True)


import numpy as np




#Scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
bats_scaled = pd.DataFrame(scaler.fit_transform(bats1[['Origin', 'Matches Played', 'Runs_20', 'Runs_21', 'Runs_22', 'BPBF_20','BPBF_21', 'BPBF_22', 'SR_20', 'SR_21', 'SR_22','Players_Total_Auction_price', 'Instagram Followers(in M)','Twitter Followers(in M)','Player of the match Awards(All seasons 2020-2022)','Played in IPL(in years)','IPL Followers (Team wise) in Instagram(Fanbase)','Age']]))
bats_scaled.columns = ['Origin', 'Matches Played', 'Runs_20', 'Runs_21', 'Runs_22', 'BPBF_20','BPBF_21', 'BPBF_22', 'SR_20', 'SR_21', 'SR_22', 'Players_Total_Auction_price', 'Instagram Followers(in M)','Twitter Followers(in M)','Player of the match Awards(All seasons 2020-2022)','Played in IPL(in years)','IPL Followers (Team wise) in Instagram(Fanbase)','Age']
bats1 = bats1.drop(['Origin', 'Matches Played', 'Runs_20', 'Runs_21', 'Runs_22', 'BPBF_20','BPBF_21', 'BPBF_22', 'SR_20', 'SR_21', 'SR_22', 'Players_Total_Auction_price', 'Instagram Followers(in M)','Twitter Followers(in M)','Player of the match Awards(All seasons 2020-2022)','Played in IPL(in years)','IPL Followers (Team wise) in Instagram(Fanbase)','Age'], axis=1)
bats_scaled.reset_index(drop=True, inplace=True)
bats1.reset_index(drop=True, inplace=True)
final = pd.concat([bats1, bats_scaled], axis=1)




'''Linear Regression model for Batsmen'''

# bats1.convert_dtypes()
# bats = pd.get_dummies(bats, drop_first = True)
#Splitting x and y variables
x = final.drop('Players_Total_Auction_price', axis=1)
y = final['Players_Total_Auction_price']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state= 2)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions on the test set
y_pred_pred = lr.predict(x_test)

coefficients = lr.coef_
variable_coefficients = dict(zip(x.columns, coefficients))
for variable, coefficient in variable_coefficients.items():
    print(f'Variable: {variable}, Coefficient: {coefficient}')
    
#Performance of the model
import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 8.436892186305851
#Create a scatter plot of predicted vs actual values
#plt.scatter(y_test, y_pred)
##Plotting the regression model
#plt.figure(figsize=(12,9))
#plt.scatter(y_test, y_pred)
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
#plt.xlabel('Actual Total Auction Price')
#plt.ylabel('Predicted total Aucttion Price')
#plt.title('Linear Regression for Batsmen with Batting Points, IPL Stats and Social Media Presence')
#plt.show()
players = bats['Player'].tolist()
Instagram_followers = bats['Instagram Followers(in M)'].tolist()
import matplotlib.pyplot as plt

# Create a scatter plot of predicted vs actual values
plt.figure(figsize=(12, 9))
plt.scatter(y_test, y_pred_pred)

for a, b, player_name, Instagram_followers in zip(y_test,y_pred_pred, players, Instagram_followers):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)



# Add axis labels and title
plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted Total Auction Price')
plt.title('Linear Regression for Batsmen with Batting Points, IPL Stats and Social Media Presence')

plt.show()




'''Linear Regression model for Batsmen excluding Social Media Presence'''

#Splitting x and y variables
x = final.drop(['Instagram Followers(in M)', 'Twitter Followers(in M)', 'IPL Followers (Team wise) in Instagram(Fanbase)'], axis=1)

y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=2)




#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)


coefficients = lr.coef_
variable_coefficients = dict(zip(x.columns, coefficients))
for variable, coefficient in variable_coefficients.items():
    print(f'Variable: {variable}, Coefficient: {coefficient}')
    

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([11.42394906, -7.39539659,  2.07405891,  4.07685908])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 10.643415590603377

#Create a scatter plot of predicted vs actual values
plt.figure(figsize=(12, 9))
plt.scatter(y_test, y_pred)

batting_points = bats['Batting_points'].tolist()
Instagram_followers = bats['Instagram Followers(in M)'].tolist()

for a, b, player_name, Instagram_followers in zip(y_test, y_pred, players, Instagram_followers):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)

# Add axis labels and title
plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted Total Auction Price')
plt.title('Linear Regression for Batsmen with Batting Points, IPL Stats and Excluding Social Media Presence')

plt.show()




'''Linear Regression for Batsman Excluding Player of the Match'''

x = final.drop(['Player of the match Awards(All seasons 2020-2022)'], axis = 1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=2)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([11.42394906, -7.39539659,  2.07405891,  4.07685908])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 10.643415590603377
plt.figure(figsize=(12, 9))
#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)


#batting_points = bats['Batting_points'].tolist()


for a, b, player_name in zip(y_test, y_pred, players):
    plt.text(a, b, f'{player_name}', fontsize=10, ha='left', va='bottom')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)

# Add axis labels and title
plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted Total Auction Price')
plt.title('Linear Regression for Batsman Excluding Player of the Match')
#plt.ylim(bottom=0)
plt.show()



'''Linear Regression for Batsman with Batting Points and Age'''


x = final.drop(['Age'], axis = 1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=2)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([11.42394906, -7.39539659,  2.07405891,  4.07685908])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 10.643415590603377

plt.figure(figsize=(12, 9))
#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
batting_points = bats['Batting_points'].tolist()


for a, b, player_name, batting_points in zip(y_test, y_pred, players, batting_points):
    plt.text(a, b, f'{player_name}, {batting_points}', fontsize=10, ha='left', va='bottom')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)

# Add axis labels and title
plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted Total Auction Price')
plt.title('Linear Regression for Batsmen Excluding Age')
plt.show()




'''Exploratory Data Analysis for Bowler data'''
bowls.dtypes #Checking for inconsistent data types and found none

print("Shape of the dataset for bowlers:", bowls.shape)
#Shape of the dataset for batsmen: (41, 38)

#Checking for outliers
bowls.boxplot(figsize=(25,15))
plt.xticks(rotation=90)
plt.show()
# We have outliers but we won't handle or replace them.

#Summary statistics for numeric columns and also for strings
num_bowl = bowls.describe()
strn_bowl = bowls.describe(include='object')

#Plotting the distribution of the target variable
sns.histplot(data=bowls, x='Players_Total_Auction_price')
plt.title('Distribution of Total Auction Price for Bowlers')
plt.show()

#Correlation matrix
plt.figure(figsize=(20,12))
corr_matrix = bowls.corr()
sns.heatmap(corr_matrix, annot=False)
plt.title('Correlation Heatmap for Bowlers')
plt.show()


mapping = {'Indian': 1, 'Overseas': 0}
bowls['Origin'] = bowls['Origin'].map(mapping)
bowls['Origin'].fillna(1, inplace=True)
bowls1 = bats.drop('Player', axis=1)
bowls1 = pd.get_dummies(bowls1, drop_first=True)


#Scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
bowls_scaled = pd.DataFrame(scaler.fit_transform(bowls1[['Origin', 'Matches Played', 'Wkts_20', 'Wkts_21', 'Wkts_22', 'Econ_20', 'Econ_21', 'Econ_22', 'Avg_20', 'Avg_21', 'Avg_22', 'Bowling_points', 'P_Bowling_points', 'Auction_Price_in_Cr_2020', 'Auction_Price_in_Cr_2021', 'Auction_Price_in_Cr_2022', 'Players_Total_Auction_price', 'Instagram Followers(in M)', 'Twitter Followers(in M)', 'Player of the match Awards(All seasons 2020-2022)', 'Played in IPL(in years)', 'Team in IPL', 'IPL Followers (Team wise) in Instagram(in M) (Fanbase)']]))
bowls_scaled.columns = ['Origin', 'Matches Played', 'Wkts_20', 'Wkts_21', 'Wkts_22', 'Econ_20', 'Econ_21', 'Econ_22', 'Avg_20', 'Avg_21', 'Avg_22', 'Bowling_points', 'P_Bowling_points', 'Auction_Price_in_Cr_2020', 'Auction_Price_in_Cr_2021', 'Auction_Price_in_Cr_2022', 'Players_Total_Auction_price', 'Instagram Followers(in M)', 'Twitter Followers(in M)', 'Player of the match Awards(All seasons 2020-2022)', 'Played in IPL(in years)', 'Team in IPL', 'IPL Followers (Team wise) in Instagram(in M) (Fanbase)']
bowls = bowls1.drop(['Origin', 'Matches Played', 'Wkts_20', 'Wkts_21', 'Wkts_22', 'Econ_20', 'Econ_21', 'Econ_22', 'Avg_20', 'Avg_21', 'Avg_22', 'Bowling_points', 'P_Bowling_points', 'Auction_Price_in_Cr_2020', 'Auction_Price_in_Cr_2021', 'Auction_Price_in_Cr_2022', 'Players_Total_Auction_price', 'Instagram Followers(in M)', 'Twitter Followers(in M)', 'Player of the match Awards(All seasons 2020-2022)', 'Played in IPL(in years)', 'Team in IPL', 'IPL Followers (Team wise) in Instagram(in M) (Fanbase)'], axis=1)
bowls_scaled.reset_index(drop=True, inplace=True)
bowls1.reset_index(drop=True, inplace=True)
final = pd.concat([bowls1, bowls_scaled], axis=1)



'''Linear Regression model for Bowlers'''

#Splitting x and y variables
x = final.drop(['Players_Total_Auction_price'], axis=1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=5)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#[-9.37096616,  6.84448698,  5.03709687, -0.53627505,  3.0861558 ,3.03595368, -1.84776626])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 7.69037011658539

#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
players = bowls['Player'].tolist()

#Plotting the regression model
plt.figure(figsize=(12,9))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
bowling_points = bowls['Bowling_points'].tolist()
Instagram_followers = bowls['Instagram Followers(in M)'].tolist()


for a, b, player_name, Instagram_followers in zip(y_test, y_pred, players,np.round(Instagram_followers, 2)):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted total Aucttion Price')
plt.title('Linear Regression for Bowlers with Bowling Points, IPL Stats and Social Media Presence')
plt.show()

'''Linear Regression model for Bowlers excluding Social Media Presence'''

#Splitting x and y variables
x = final.drop(['Instagram Followers(in M)', 'Twitter Followers(in M)', 'IPL Followers (Team wise) in Instagram(Fanbase)'], axis = 1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=3)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([-0.94979999,  0.35464595,  3.10868154,  4.33572388])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 7.447723401819984

#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)

bowling_points = bowls['Bowling_points'].tolist()
Instagram_followers = bowls['Instagram Followers(in M)'].tolist()
#Plotting the regression model
plt.figure(figsize=(12,9))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)



for a, b, player_name, Instagram_followers in zip(y_test, y_pred, players,np.round(Instagram_followers, 2)):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted total Aucttion Price')
plt.title('Linear Regression for Bowlers with Bowling Points, IPL Excluding Social Media Presence')
plt.show()


'''Linear Regression for Bowlers Excluding Social media status man of the match and Played in IPL'''



x = final.drop(['Player of the match Awards(All seasons 2020-2022)'], axis = 1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=5)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([-0.94979999,  0.35464595,  3.10868154,  4.33572388])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 7.447723401819984

#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)

bowling_points = bowls['Bowling_points'].tolist()
Instagram_followers = bowls['Instagram Followers(in M)'].tolist()

#Plotting the regression model
plt.figure(figsize=(12,9))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)


for a, b, player_name, Instagram_followers in zip(y_test, y_pred, players,np.round(Instagram_followers, 2)):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted total Aucttion Price')
plt.title('Linear Regression for Bowlers Excluding man of the match')
plt.show()



'''Linear Regression for Bowler with Batting Points and Age'''


x = final.drop(['Age'], axis = 1)
y = final['Players_Total_Auction_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=5)

#Scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the linear regression model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_scaled)

lr.coef_
#([-0.94979999,  0.35464595,  3.10868154,  4.33572388])

#Performance of the model
mse = mean_squared_error(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(mse))
#Root Mean Squared Error: 7.447723401819984

players = bowls['Player'].tolist()
bowling_points = bowls['Bowling_points'].tolist()
Instagram_followers = bowls['Instagram Followers(in M)'].tolist()
#Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)



#Plotting the regression model
plt.figure(figsize=(12,9))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)


for a, b, player_name, Instagram_followers in zip(y_test, y_pred, players,np.round(Instagram_followers, 2)):
    plt.text(a, b, f'{player_name}, {Instagram_followers} M', fontsize=10, ha='left', va='bottom')

plt.xlabel('Actual Total Auction Price')
plt.ylabel('Predicted total Aucttion Price')
plt.title('Linear Regression for Bowlers Excluding  Age')
plt.show()










