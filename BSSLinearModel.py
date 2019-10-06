import pandas as pd
import numpy as np

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle

chicago_bike_data_Q1= pd.read_csv('datasets/Divvy_Trips_2018_Q1.csv')
chicago_bike_data_Q2= pd.read_csv('datasets/Divvy_Trips_2018_Q2.csv')
chicago_bike_data_Q3= pd.read_csv('datasets/Divvy_Trips_2018_Q3.csv')
chicago_bike_data_Q4= pd.read_csv('datasets/Divvy_Trips_2018_Q4.csv')

chicago_bike_data_Q1 = chicago_bike_data_Q1.rename(columns={"03 - Rental Start Station Name":"from_station_name",
                                  "02 - Rental End Station Name":"to_station_name","01 - Rental Details Rental ID":"trip_id"})

data_groupby_day_out_Q1 = pd.DataFrame(chicago_bike_data_Q1.groupby(['from_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q1 = pd.DataFrame(chicago_bike_data_Q1.groupby(['to_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q1= data_groupby_day_out_Q1.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q1 = data_groupby_day_in_Q1.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q1=pd.merge(data_groupby_day_out_Q1,data_groupby_day_in_Q1,on=['Station Name'],how='outer')

data_groupby_day_out_Q2 = pd.DataFrame(chicago_bike_data_Q2.groupby(['from_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q2 = pd.DataFrame(chicago_bike_data_Q2.groupby(['to_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q2= data_groupby_day_out_Q2.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q2 = data_groupby_day_in_Q2.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q2=pd.merge(data_groupby_day_out_Q2,data_groupby_day_in_Q2,on=['Station Name'],how='outer')

data_groupby_day_out_Q3 = pd.DataFrame(chicago_bike_data_Q3.groupby(['from_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q3 = pd.DataFrame(chicago_bike_data_Q3.groupby(['to_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q3= data_groupby_day_out_Q3.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q3 = data_groupby_day_in_Q3.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q3=pd.merge(data_groupby_day_out_Q3,data_groupby_day_in_Q3,on=['Station Name'],how='outer')

data_groupby_day_out_Q4 = pd.DataFrame(chicago_bike_data_Q4.groupby(['from_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q4 = pd.DataFrame(chicago_bike_data_Q4.groupby(['to_station_name'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q4= data_groupby_day_out_Q4.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q4 = data_groupby_day_in_Q4.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q4=pd.merge(data_groupby_day_out_Q4,data_groupby_day_in_Q4,on=['Station Name'],how='outer')

new_df = pd.concat([new_df_Q1,new_df_Q2,new_df_Q3,new_df_Q4])

top_rentals = new_df.sort_values(['Number Of Outgoing Trips'], ascending=False).iloc[0:10,0:2].reset_index()

chicago_bike_data_Q1['day'] = [int(str(starttime).split(" ")[0].split("-")[2]) for starttime in chicago_bike_data_Q1['01 - Rental Details Local Start Time']]
chicago_bike_data_Q1['month'] = [int(str(starttime).split(" ")[0].split("-")[1]) for starttime in chicago_bike_data_Q1['01 - Rental Details Local Start Time']]
chicago_bike_data_Q1['hour'] = [int(str(starttime).split(" ")[1].split(":")[0]) for starttime in chicago_bike_data_Q1['01 - Rental Details Local Start Time']]

chicago_bike_data_Q2['day'] = [int(str(starttime).split(" ")[0].split("-")[2]) for starttime in chicago_bike_data_Q2['start_time']]
chicago_bike_data_Q2['month'] = [int(str(starttime).split(" ")[0].split("-")[1]) for starttime in chicago_bike_data_Q2['start_time']]
chicago_bike_data_Q2['hour'] = [int(str(starttime).split(" ")[1].split(":")[0]) for starttime in chicago_bike_data_Q2['start_time']]

chicago_bike_data_Q3['day'] = [int(str(starttime).split(" ")[0].split("-")[2]) for starttime in chicago_bike_data_Q3['start_time']]
chicago_bike_data_Q3['month'] = [int(str(starttime).split(" ")[0].split("-")[1]) for starttime in chicago_bike_data_Q3['start_time']]
chicago_bike_data_Q3['hour'] = [int(str(starttime).split(" ")[1].split(":")[0]) for starttime in chicago_bike_data_Q3['start_time']]

chicago_bike_data_Q4['day'] = [int(str(starttime).split(" ")[0].split("-")[2]) for starttime in chicago_bike_data_Q4['start_time']]
chicago_bike_data_Q4['month'] = [int(str(starttime).split(" ")[0].split("-")[1]) for starttime in chicago_bike_data_Q4['start_time']]
chicago_bike_data_Q4['hour'] = [int(str(starttime).split(" ")[1].split(":")[0]) for starttime in chicago_bike_data_Q4['start_time']]

data_groupby_day_out_Q1 = pd.DataFrame(chicago_bike_data_Q1.groupby(['from_station_name', 'month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q1 = pd.DataFrame(chicago_bike_data_Q1.groupby(['to_station_name','month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q1= data_groupby_day_out_Q1.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q1 = data_groupby_day_in_Q1.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q1=pd.merge(data_groupby_day_out_Q1,data_groupby_day_in_Q1,on=['Station Name','month','day','hour'],how='outer')
new_df_Q1 = new_df_Q1.sort_values(['month','day','hour'], ascending=True).reset_index()

data_groupby_day_out_Q2 = pd.DataFrame(chicago_bike_data_Q2.groupby(['from_station_name', 'month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q2 = pd.DataFrame(chicago_bike_data_Q2.groupby(['to_station_name','month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q2= data_groupby_day_out_Q2.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q2 = data_groupby_day_in_Q2.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q2=pd.merge(data_groupby_day_out_Q2,data_groupby_day_in_Q2,on=['Station Name','month','day','hour'],how='outer')
new_df_Q2 = new_df_Q2.sort_values(['month','day','hour'], ascending=True).reset_index()

data_groupby_day_out_Q3 = pd.DataFrame(chicago_bike_data_Q3.groupby(['from_station_name', 'month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q3 = pd.DataFrame(chicago_bike_data_Q3.groupby(['to_station_name','month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q3= data_groupby_day_out_Q3.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q3 = data_groupby_day_in_Q3.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q3=pd.merge(data_groupby_day_out_Q3,data_groupby_day_in_Q3,on=['Station Name','month','day','hour'],how='outer')
new_df_Q3 = new_df_Q3.sort_values(['month','day','hour'], ascending=True).reset_index()

data_groupby_day_out_Q4 = pd.DataFrame(chicago_bike_data_Q4.groupby(['from_station_name', 'month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_in_Q4 = pd.DataFrame(chicago_bike_data_Q4.groupby(['to_station_name','month','day','hour'])['trip_id'].count()).reset_index()
data_groupby_day_out_Q4= data_groupby_day_out_Q4.rename(columns={"trip_id": "Number Of Outgoing Trips",
                                                    "from_station_name":"Station Name"})
data_groupby_day_in_Q4 = data_groupby_day_in_Q4.rename(columns={"trip_id": "Number Of Incoming Trips",
                                                    "to_station_name":"Station Name"})
new_df_Q4=pd.merge(data_groupby_day_out_Q4,data_groupby_day_in_Q4,on=['Station Name','month','day','hour'],how='outer')
new_df_Q4 = new_df_Q4.sort_values(['month','day','hour'], ascending=True).reset_index()

new_df = pd.concat([new_df_Q1,new_df_Q2,new_df_Q3,new_df_Q4])

new_df['Number Of Outgoing Trips'] = new_df['Number Of Outgoing Trips'].fillna(0)
new_df['Number Of Incoming Trips'] = new_df['Number Of Incoming Trips'].fillna(0)

def get_processed_station_data(new_df, station_name):
    new_single_station_df = new_df.loc[new_df['Station Name']==station_name]
    new_single_station_df = new_single_station_df.sort_values(['month','day'], ascending=True)
    for month in set(new_single_station_df['month']):
        new_single_station_month_df = new_single_station_df.loc[new_single_station_df['month']==month]
        for day in set(new_single_station_month_df['day']):
            new_single_station_day_df = new_single_station_month_df.loc[new_single_station_month_df['day']==day]
            list_of_hours = new_single_station_day_df['hour']
            for i in range(0,24):
                if(i not in set(list_of_hours)):
                    app_df = pd.DataFrame([(month,day,i,0)], columns=['month','day','hour','Number Of Outgoing Trips'])
                    new_single_station_df = new_single_station_df.append(app_df,sort=True)
    new_single_station_df = new_single_station_df.sort_values(['month','day','hour'], ascending=True).reset_index()
    new_single_station_df = new_single_station_df.iloc[:,[4,5,2]]
    new_single_station_df= new_single_station_df.iloc[:,[2]]
    return new_single_station_df

def random_forest_regression(xTrain, xTest, yTrain, yTest):
    rf_regressor = RandomForestRegressor(n_estimators=100,criterion='mse',max_features=1, oob_score=True)
    rf_regressor.fit(xTrain, yTrain)
    return rf_regressor

station_data = pd.DataFrame(columns=['Station Name',
                                    'Random Forest MSE',
                                    'Random Forest MEAN AE',
                                    'Random Forest R2 SCORE',
                                    'Random Forest MEDIAN AE'
                                     ])
test_data_size = 1
for station_name in top_rentals['Station Name'].unique():
    single_station_df = get_processed_station_data(new_df, station_name)
    training_set = single_station_df.iloc[0:single_station_df.shape[0]-test_data_size,0:1].values
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    xTrain = []
    yTrain = []
    for i in range(24, training_set_scaled.shape[0]):
        xTrain.append(training_set_scaled[i-24:i,0])
        yTrain.append(training_set_scaled[i,0])
    xTrain, yTrain = np.array(xTrain),np.array(yTrain)
    inputs = single_station_df.iloc[single_station_df.shape[0]-24-test_data_size:single_station_df.shape[0],0:1].values
    inputs = sc.transform(inputs)
    yTest = single_station_df.iloc[single_station_df.shape[0]-test_data_size:single_station_df.shape[0],0:1].values
    xTest = []
    for i in range(24,inputs.shape[0]):
        xTest.append(inputs[i-24:i,0])
    xTest = np.array(xTest)
    print(xTest.shape)
    print(xTest)
    model = random_forest_regression(xTrain, xTest, yTrain, yTest)
    print(sc.inverse_transform(model.predict(xTest.reshape(1,24)).reshape(-1,1)))
    pickle.dump(model,open('bss_model.pkl','wb'))