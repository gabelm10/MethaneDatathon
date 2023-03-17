from sklearn.model_selection import train_test_split
import pandas as pd

ROOT_PATH = "C://Users//gabri//Documents//MethaneDatathon//"

# Read preprocessed test data
data_df = pd.read_csv(ROOT_PATH + "data_pixel_counts.csv")

# Feature engineering
percent_error = 0.1
col_names=["R", "Y", "G", "B", "Total",
           "R2", "Y2", "G2", "B2",
           "R3", "Y3", "G3", "B3",
           "R/Tot", "Y/Tot", "G/Tot", "B/Tot",
           "Rinv", "Yinv", "Ginv", "Binv", 
           "Rsqrt", "Ysqrt", "Gsqrt", "Bsqrt"]
lst = []
for i in col_names:
    lst.append(0)
x = pd.DataFrame(columns=col_names)
y = []
for index, row in data_df.iterrows():
    R = row['red_pixels']
    Y = row['yellow_pixels']
    G = row['green_pixels']
    B = row['blue_pixels']
    if R < 0:
        R = 0
    if Y < 0:
        Y = 0
    if G < 0:
        G = 0
    if B < 0:
        B = 0
    Total = 4 * R + 3 * Y + 2 * G + B
    lst[0] = R
    lst[1] = Y
    lst[2] = G
    lst[3] = B
    lst[4] = Total
    lst[5] = R**2
    lst[6] = Y**2
    lst[7] = G**2
    lst[8] = B**2
    lst[9] = R**3
    lst[10] = Y**3
    lst[11] = G**3
    lst[12] = B**3
    if Total > 0: # Else, it is already set to zero
        lst[13] = R / Total
        lst[14] = Y / Total
        lst[15] = G / Total
        lst[16] = B / Total
    lst[17] = R**0.5
    lst[18] = Y**0.5
    lst[19] = G**0.5
    lst[20] = B**0.5
    x.loc[len(x)] = lst
    x.loc[len(x)] = lst
    x.loc[len(x)] = lst
    
    y.append(max(0.0,row['emission'] - percent_error * row['emission_uncertainty']))
    y.append(row['emission'])
    y.append(row['emission'] + percent_error * row['emission_uncertainty'])
# Make sure there are no nan
x = x.fillna(0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Best random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0, max_depth= 25, max_features='sqrt', criterion='squared_error')

# fit the regressor with x and y data
regressor.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import mean_squared_error
prediction = regressor.predict(X_val)
mse = mean_squared_error(y_val, prediction)
rmse = mse**.5
print('Validation RMSE:', rmse)






# Read preprocessed test data
test_df = pd.read_csv(ROOT_PATH + "test_pixel_count.csv")

x_pred = pd.DataFrame(columns=col_names)
for index, row in test_df.iterrows():
    R = row['red_pixels']
    Y = row['yellow_pixels']
    G = row['green_pixels']
    B = row['blue_pixels']
    if R < 0:
        R = 0
    if Y < 0:
        Y = 0
    if G < 0:
        G = 0
    if B < 0:
        B = 0
    Total = 4 * R + 3 * Y + 2 * G + B
    lst[0] = R
    lst[1] = Y
    lst[2] = G
    lst[3] = B
    lst[4] = Total
    lst[5] = R**2
    lst[6] = Y**2
    lst[7] = G**2
    lst[8] = B**2
    lst[9] = R**3
    lst[10] = Y**3
    lst[11] = G**3
    lst[12] = B**3
    if Total > 0: # Else, it is already set to zero
        lst[13] = R / Total
        lst[14] = Y / Total
        lst[15] = G / Total
        lst[16] = B / Total
    lst[17] = R**0.5
    lst[18] = Y**0.5
    lst[19] = G**0.5
    lst[20] = B**0.5
    x_pred.loc[len(x_pred)] = lst
# Make sure there are no nan
x_pred = x_pred.fillna(0)

# Get results
MethaneRate_Actual = regressor.predict(x_pred)
print(MethaneRate_Actual)

# Save results
total_results = len(MethaneRate_Actual)
source_ID = ["01A", "02A","03A","04A","05A","06A","07A","08A","09A","10A",
             "11A","12A","13A","14A","15A","16A","17A","18A","19A","20A",]

results_df = pd.DataFrame(list(zip(source_ID, MethaneRate_Actual)), columns =['source_ID', 'MethaneRate_Actual'])

results_df.to_csv(ROOT_PATH + 'results.csv', index=False)