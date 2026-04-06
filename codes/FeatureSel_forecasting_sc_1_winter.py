# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from neuralforecast.auto import NHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MSE
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures,LagFeatures
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sktime.transformations.series.fourier import FourierFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
import holidays
import xgboost
from sklearn.ensemble import RandomForestRegressor


# %%
# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the GPU being used
print("Current device:", torch.cuda.current_device())

# Print the name of the GPU
print("GPU in use:", torch.cuda.get_device_name(torch.cuda.current_device()))

# %%
def CreateWorkHourFeature(input_data):
    """
    Receives as input a DataFrame or Series and outputs a DataFrame with the working hours during the day.
    When the day of the week is larger than 4, it is considered a weekend (1), otherwise, it's a workday (0).
    During workdays and between 8:00 and 17:00, it is considered a working hour.

    Parameters:
    input_data (DataFrame or Series): Input data with a DatetimeIndex.

    Returns:
    DataFrame: DataFrame with the added "WorkingHour_flag" column.
    """
    if isinstance(input_data, pd.Series):
        input_df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Input must be a DataFrame or Series.")

    assert isinstance(input_df.index, pd.DatetimeIndex), "Index must be a datetime index."

    input_df["dayOfWeek"] = input_df.index.dayofweek
    input_df.loc[input_df["dayOfWeek"] > 4, "weekendFlag"] = 1
    input_df.loc[input_df["dayOfWeek"] < 5, "weekendFlag"] = 0
    input_df["hour"] = input_df.index.hour
    input_df["WorkingHour_flag"] = 0
    input_df.loc[((input_df["hour"] > 8) & (input_df["hour"] < 17) & (input_df["weekendFlag"] == 0)), "WorkingHour_flag"] = 1
    input_df.drop(["hour", "dayOfWeek", "weekendFlag"], axis=1, inplace=True)

    return input_df


def ListCreatorFlagger(df, substrings=['flag', 'cos', 'sin','day_of_week', 'day_of_month', 'weekend', 'days_in_month', 'hour', 'minute']):
    """
    A function that separates the columns containing specified substrings from those that don't.
    df is the dataframe in question and the substring is a list.
    """
    flag_columns = [col for col in df.columns if any(substring in col for substring in substrings)]

    if not flag_columns:
        print("No columns with the specified substrings found.")
        return None, None

    non_flag_columns = [col for col in df.columns if col not in flag_columns]

    return non_flag_columns, flag_columns


def HolidayFeatureCreator(input_data):
    """
    Receives as input a DataFrame or Series and creates a column named "Holidays_flag" with 1 if there is a holiday and with 0 if no holidays exist.
    Holidays derived from Germany.
    """
    if isinstance(input_data, pd.Series):
        input_df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Input must be a DataFrame or Series.")

    assert isinstance(input_df.index, pd.DatetimeIndex), "Index must be a datetime index."

    national_holidays_all = holidays.DE(years=[2014,2015,2016,2017,2018,2019,2020, 2021, 2022, 2023, 2024, 2025, 2026]).items()
    national_holidays = [items[0] for items in national_holidays_all]  # this is a list
    
    # Create a new column for holidays flag
    input_df["Holidays_flag"] = 0
    
    # Iterate over the index and set holiday flag to 1 if the date matches any national holiday
    for index_date in input_df.index:
        if index_date.date() in national_holidays:
            input_df.at[index_date, "Holidays_flag"] = 1
    
    return input_df




def TimeRelatedFeatureConstructor(df):
  """
  Works only in a dataframe as input: run the other functions first.
  Extracts time-related features
  """
  TimeFeaturesToExtract=["day_of_week","weekend","hour","minute",] #consider to add more
  dtfs=DatetimeFeatures(variables="index", features_to_extract=TimeFeaturesToExtract, drop_original=False)
  df=dtfs.fit_transform(df)

  CyclicalFeaturesToExtract=["day_of_week","hour","minute",]
  cyclical_dtfs=CyclicalFeatures(variables=CyclicalFeaturesToExtract,drop_original=False)
  df=cyclical_dtfs.fit_transform(df)
  return df


def FourierFeatureConstructor(df,granularity,fourier_terms_list):
    
    number_part = ''.join(filter(str.isdigit, granularity))
    number_int = int(number_part)
    minutes4hour=60/number_int
    
    Fourier_Transformer=FourierFeatures(
        sp_list=[minutes4hour, 24*minutes4hour,24*7*minutes4hour,24*30*minutes4hour],   # hourly, 24 is daily seasonality *12 because for 60 min we have 12 intervals,  and 24*7 is weekly seasonality
        fourier_terms_list=fourier_terms_list,
        freq=granularity, #not necessery
        keep_original_columns=True,

    )

    Fourier_Transformer.fit(df)
    df=Fourier_Transformer.transform(df)
    return df

def WindowFeaturesConstructor(df, granularity, ListWithNoFlags):
    """
    This is a function that makes a list of 4 window features starting from double the granularity and following by doubling the previous value
    """
    number_part = ''.join(filter(str.isdigit, granularity))
    number_int = int(number_part)
    double_granularity = 2 * number_int
    time_intervals = [double_granularity]
    
    # Calculate subsequent values
    for i in range(3):
        time_intervals.append(time_intervals[-1] * 2)
    
    windowlist = [interval // number_int for interval in time_intervals]  # Corrected division
    functionsList = ["mean", "std"]
    WindownFeatureTransformer = WindowFeatures(variables=ListWithNoFlags,
                                               functions=functionsList,
                                               window=windowlist,
                                               freq=granularity,
                                               drop_original=False)

    df = WindownFeatureTransformer.fit_transform(df)
    return df

def ExpandingWindowFeatureConstructor(df,ListWithNoFlags):
  functionsList=["mean","std"]
  frequency = pd.infer_freq(df.index) #infer the frequency from the dataframe
  ExpandingWindownFeatureTransformer=ExpandingWindowFeatures(variables=ListWithNoFlags,
                                                           functions=functionsList,
                                                           freq=frequency, #I put the freq to shift it down! but now it is performed automatically!
                                                           drop_original=False)
  df=ExpandingWindownFeatureTransformer.fit_transform(df)
  return df

def WeightedLinearFeatureMaker(df,ListWithNoFlags,granularity):
  """
  This is a function that takes the original DF and modifies the continious value columns
  Inputs: Dataframe, List of columns that are continous values, daily window to slide, weights of the values
  """
  number_part = ''.join(filter(str.isdigit, granularity))
  Minutedensity=int(number_part)
  Window=int((60/Minutedensity)*24) #288 means a daily window
  weights=np.arange(1,Window+1)

  # if i had hourly data then i would have had np.arange(1,24*7) for a weekly window

  def weighted_mean (x,weights):
    return (weights*x).sum()/weights.sum()

  def weighted_std(x,weights):
    mean_w= weighted_mean(x, weights)
    var_w= (weights* (x-mean_w)**2).sum()/weights.sum()
    return np.sqrt(var_w)

  # LETS make the weighted mean column
  for i in ListWithNoFlags:
    result=(
        df[i]
        .rolling(window=Window) #here we pick a window size. Needs to be the same as the len(weights)
        .apply(weighted_mean, args=(weights,))
        .shift(1)#shift by 1 to avoid data leakage
        .to_frame()#convert series to df
        )

    result.columns=[str(i)+"_weighted_"+str(Window)+"_mean"]
    df=df.join(result)

  for i in ListWithNoFlags:
    result=(
        df[i]
        .rolling(window=Window) #here we pick a window size. Needs to be the same as the len(weights)
        .apply(weighted_std, args=(weights,))
        .shift(1)#shift by 1 to avoid data leakage
        .to_frame()#convert series to df
        )

    result.columns=[str(i)+"_weighted_"+str(Window)+"_std"]
    df=df.join(result)
  return df

def ExpWeightMeanMaker(df,ListWithNoFlags,granularity):
  """
  This is a function that makes exp weighted average with a sliding window approach
  """
  number_part = ''.join(filter(str.isdigit, granularity))
  Minutedensity=int(number_part)
  Window=int((60/Minutedensity)*24) #288 means a daily window
  
  def exp_weights(alpha,window_size):
    """
    a function to calculate the weights for every single component of our sliding windown
    """
    weights=np.ones(window_size) #initializing weights
    for ix in range(window_size):
      weights[ix]=(1-alpha)**(window_size-1-ix)
    return weights

  def exp_weighted_mean(x):
    """
    a functions that calculates the exp weigted mean
    """

    weights=exp_weights(alpha=0.05, window_size=len(x)) # HERE WE SET THE ALPHA
    return (weights*x).sum()/weights.sum()
  
  for i in ListWithNoFlags:
    result=(
        df[i]
        .rolling(window=int(Window))
        .agg([exp_weighted_mean])
        .shift(1)
    )


    result.columns=[str(i)+"_Exp_weighted_"+str(Window)+"_SL.win"]
    df=df.join(result)
  return df

def WeightedExponentialExpandingWindow(df,ListWithNoFlags,alpha):
  """
  This is a funtion that takes as input the df,a list of continuous values and the alpha.
  Outputs: all continuous features on the df that are "mean" and "std"
  """

  for i in ListWithNoFlags:
    df[[str(i)+"_ewm_mean_expanding.win",str(i)+"ewm_std_expanding.win"]]= (
                                              df[i].ewm(alpha=alpha).
                                              agg(["mean","std"])
                                              .shift(1)
                                            )
  return df

def FeatureLagger(df,ListOfFeatures,granularity,PredictionHorizon):

    time_intervals = []
    number_part = ''.join(filter(str.isdigit, granularity))
    Minutedensity=int(number_part)
    end_in_day=int((PredictionHorizon)/(Minutedensity))
    for i in range(1, 1+end_in_day):  # 24 hours * 60 minutes / 15 minutes = 96 intervals
        time_intervals.append(f"{i * 15}min")

    lag_transformer= LagFeatures(variables=ListOfFeatures,
                                freq=time_intervals,
                                drop_original=False) #make a lagger transformer drop all original features

    df=lag_transformer.fit_transform(df) # transform the features to DF joined
    return df

# %%
def ErrorCalculator(name, y_true, y_pred):
    errors = {"Pipelines": name,
              "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
              "MAE": mean_absolute_error(y_true, y_pred),
              "MSE": mean_squared_error(y_true, y_pred),
              
              "PMAE": peak_MAE(y_true, y_pred),
              "PMSE": peak_MSE(y_true, y_pred),
              "PRMSE": peak_RMSE(y_true, y_pred),
              
              "VMAE": valley_MAE(y_true, y_pred),
              "VMSE": valley_MSE(y_true, y_pred),
              "VRMSE": valley_RMSE(y_true, y_pred),
              
              "xMAE": extreme_MAE(y_true, y_pred),
              "xMSE": extreme_MSE(y_true, y_pred),
              "xRMSE": extreme_RMSE(y_true, y_pred),
             }
    return errors

def peak_MAE(y_true, y_pred, threshold=2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true > threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    mse = mean_absolute_error(filtered_real_values, filtered_predictions)
    return mse

def peak_MSE(y_true, y_pred, threshold=2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true > threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    mse = mean_squared_error(filtered_real_values, filtered_predictions)
    return mse

def peak_RMSE(y_true, y_pred, threshold=2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true > threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    rmse = np.sqrt(mean_squared_error(filtered_real_values, filtered_predictions))
    return rmse

def valley_MAE(y_true, y_pred, lower_threshold=-2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true < lower_threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    mse = mean_absolute_error(filtered_real_values, filtered_predictions)
    return mse

def valley_MSE(y_true, y_pred, lower_threshold=-2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true < lower_threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    mse = mean_squared_error(filtered_real_values, filtered_predictions)
    return mse

def valley_RMSE(y_true, y_pred, lower_threshold=-2000):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    filtered_indices = y_true < lower_threshold
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    rmse = np.sqrt(mean_squared_error(filtered_real_values, filtered_predictions))
    return rmse
def extreme_MAE(y_true, y_pred, lower_threshold=-2000, upper_threshold=2000):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter the indices where the true values are either below the lower threshold or above the upper threshold
    filtered_indices = (y_true < lower_threshold) | (y_true > upper_threshold)
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Absolute Error (MAE) for the filtered data
    mae = mean_absolute_error(filtered_real_values, filtered_predictions)
    return mae

def extreme_MSE(y_true, y_pred, lower_threshold=-2000, upper_threshold=2000):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter the indices where the true values are either below the lower threshold or above the upper threshold
    filtered_indices = (y_true < lower_threshold) | (y_true > upper_threshold)
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Mean Squared Error (MSE) for the filtered data
    mse = mean_squared_error(filtered_real_values, filtered_predictions)
    return mse

def extreme_RMSE(y_true, y_pred, lower_threshold=-2000, upper_threshold=2000):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter the indices where the true values are either below the lower threshold or above the upper threshold
    filtered_indices = (y_true < lower_threshold) | (y_true > upper_threshold)
    
    # Filter both predictions and true values based on these indices
    filtered_predictions = y_pred[filtered_indices]
    filtered_real_values = y_true[filtered_indices]
    
    # Calculate the Root Mean Squared Error (RMSE) for the filtered data
    rmse = np.sqrt(mean_squared_error(filtered_real_values, filtered_predictions))
    return rmse


# %%
#example https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html

# %%

# Check if CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Check the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Print the name of the GPU
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name()}")
# Set device to GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using device: {device}")

# %%
df = pd.read_csv(r"D:\dimitry\Net load forecasting for NZEB\inputs\Clean_NISThomeAllfeatures.csv",index_col=0, parse_dates=[0])

#all dataframes should have these columns ['unique_id', 'ds', 'y']
# in this case the y is the price and the rest are covariates

# %% [markdown]
# # setting up the scenario parameters

# %%
#Calculate NetLoad!
OriginalFeatures=["NetLoad"]
scenario="1"
season="winter" #for winter it works!
OutputPath = r"D:\dimitry\Net load forecasting for NZEB\outputs\Scenario" + scenario + "\\" + season

granularity="15min"
prediction_horizon="1440min"

number_part_hor = ''.join(filter(str.isdigit, prediction_horizon))
PredictionHorizon=int(number_part_hor)
number_part = ''.join(filter(str.isdigit, granularity))
Minutedensity=int(number_part)
fourier_terms_list=[2,2,2,2]
prediction_horizon_steps=PredictionHorizon//Minutedensity # this is 96

# %%

if season=="summer":
    df = df[df.index.month.isin([3, 4, 5, 6])]
    #df = df[((df.index.month == 7) & (df.index.day <= 15)) |  # First half of July
    #    ((df.index.month == 3) & (df.index.day > 15)) |   # Last half of March
    #    (df.index.month.isin([4, 5, 6]))]                # April, May, June

    #df = df.sort_index()  # Ensure proper chronological order
elif season=="winter":
    df=df[df.index.month.isin([9, 10, 11, 12])]

#keep the features that are needed for each scenario
if scenario=="0":
    df["NetLoad"]=df["Total_load"]-df["SolarPower"]
    df=df[["NetLoad"]]

"""if scenario=="1":
    #forecasting Net load only with features from the past
    df["NetLoad"]=df["Total_load"]-df["SolarPower"]
    df=df[["NetLoad","T_out","Irradiance"]]
    OriginalFeatures=["NetLoad","T_out","Irradiance"]    """


if scenario=="1":
    #forecasting Net load only with features from the past
    df["NetLoad"]=df["Total_load"]-df["SolarPower"]
    df=df[["NetLoad","T_out","Irradiance","Wind_speed"]]
    OriginalFeatures=["NetLoad","T_out","Irradiance","Wind_speed"]


if scenario=="2":
    #forecasting Net load with features from the past and future
    df["NetLoad"]=df["Total_load"]-df["SolarPower"]
    df=df[["NetLoad","T_out","Irradiance","Wind_speed"]]
    
if scenario=="3":
    #forecasting household load and SolarPower with features from the past. Then combine them to get the NetLoad
    df=df[["Total_load","T_out","Irradiance","Wind_speed","SolarPower"]]
    
if scenario=="4":
    #forecasting household load with features from the past and future. Then combine them to get the NetLoad 
    df=df[["Total_load","T_out","Irradiance","Wind_speed","SolarPower"]]

# %%
#Positive and negative load
df["NetLoad_flag"]=0 #zero for negativeload
df.loc[((df["NetLoad"]>=0)),"NetLoad_flag"]=1 #1 if it is positibe load

# Daily peaks
df["Small_peak_flag"] = 0  # Initialize with 0
df.loc[df["NetLoad"] >= 2000, "Small_peak_flag"] = 1  # Set to 1 where NetLoad is above 2000


# Weekly peaks
df["Medium_peak_flag"] = 0  # Initialize with 0
df.loc[df["NetLoad"] >= 4000, "Medium_peak_flag"] = 1  # Set to 1 where NetLoad is above 4000

# Monthly peaks
df["Large_peak_flag"] = 0  # Initialize with 0
df.loc[df["NetLoad"] >= 5500, "Large_peak_flag"] = 1  # Set to 1 where NetLoad is above 5500


# Daily valleys
df["Small_valley_flag"] = 0
df.loc[df["NetLoad"] <= -2000, "Small_valley_flag"] = 1 # Set daily_valley to 1 where NetLoad is below -2000

# Weekly valleys
df["Medium_valley_flag"] = 0
df.loc[df["NetLoad"] <= -4000, "Medium_valley_flag"] = 1 # Set daily_valley to 1 where NetLoad is below -2000

# Monthly valleys
df["Large_valley_flag"] = 0
df.loc[df["NetLoad"] <= -5500, "Large_valley_flag"] = 1 # Set daily_valley to 1 where NetLoad is below -2000

# %%
df=HolidayFeatureCreator(df)
df=CreateWorkHourFeature(df)
df=TimeRelatedFeatureConstructor(df)
df=FourierFeatureConstructor(df,granularity,fourier_terms_list)
ListWithNoFlags,ListWithFlags=ListCreatorFlagger(df)
df=WindowFeaturesConstructor(df,granularity,ListWithNoFlags)
df=ExpandingWindowFeatureConstructor(df,ListWithNoFlags)
df=WeightedLinearFeatureMaker(df,ListWithNoFlags,granularity)
df=ExpWeightMeanMaker(df,ListWithNoFlags,granularity)
df=WeightedExponentialExpandingWindow(df,ListWithNoFlags,0.5)
df=FeatureLagger(df,OriginalFeatures,granularity,PredictionHorizon)

# %%
#make the target
df["NetLoad_+lag"+str(PredictionHorizon)]=df["NetLoad"].shift(int(-PredictionHorizon/Minutedensity))

target="NetLoad_+lag"+str(PredictionHorizon)
df.dropna(inplace=True)

# %%
# Step 1: Determine the number of rows in the DataFrame
total_size = len(df)

# Step 2: Define the number of rows for validation and test sets
validation_test_size = 96*7*2

# Step 3: Split the data
DF_training = df.iloc[:total_size - 2 * validation_test_size]
DF_validation = df.iloc[total_size - 2 * validation_test_size:total_size - validation_test_size]
DF_test = df.iloc[total_size - validation_test_size:]

# Step 4: Verify the sizes of each set
print("Training set size:", len(DF_training))
print("Validation set size:", len(DF_validation))
print("Test set size:", len(DF_test))



# %% [markdown]
# # XGboost for validation

# %%

# Define the RandomForestRegressor
"""
regressor = xgboost.XGBRegressor(
    n_estimators=500,  # Number of boosting rounds (equivalent to number of trees)
    random_state=42,   # Ensure reproducibility
    tree_method='gpu_hist',  # Use GPU-accelerated tree construction
    predictor='gpu_predictor',  # Use GPU for prediction
)
"""
regressor = xgboost.XGBRegressor(
    random_state=42,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
)
X_train=DF_training.drop([target],axis=1) # df, all features except target in a dataframe for the first period
X_test=DF_validation.drop([target],axis=1) #df , all feature except target (same as before but for the next available period period)
y_train=DF_training[target] #one column df, of the target for the first period
y_test=DF_validation[target] #one column df of the target for the next period

regressor.fit(X_train , y_train)

predicted_val = regressor.predict(X_test)

# %%
len(X_train.columns)

# %%
name="XGBM_val"
d=ErrorCalculator(name,y_test,predicted_val)
d['features'] = len(X_train.columns)
print(d)

# %%
d

# %%
#save the errors
df_errors = pd.DataFrame(data={k: [v] for k, v in d.items()})

df_errors = df_errors.set_index('Pipelines')
df_errors

Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
Joined_DF_FilePath=os.path.join(OutputPath,Results_DF_FileName)
df_errors.to_csv(Joined_DF_FilePath)

# %%
df_errors

# %%
predicted_DF_split = pd.DataFrame(predicted_val , index=DF_validation.index, columns=["XGBM_val_AF"])

predicted_DF_split = predicted_DF_split.join(y_test)

predicted_DF_split=predicted_DF_split.loc[DF_validation.index]

predicted_DF_split.columns = ["XGBM_val_AF", "NetLoad"]


#predicted_DF_split.plot()

# %% [markdown]
# # Xgboost for test

# %%
DF_training_and_DF_validation=pd.concat([DF_training,DF_validation])

X_train=DF_training_and_DF_validation.drop([target],axis=1) # df, all features except target in a dataframe for the first period
X_test=DF_test.drop([target],axis=1) #df , all feature except target (same as before but for the next available period period)
y_train=DF_training_and_DF_validation[target] #one column df, of the target for the first period
y_test=DF_test[target] #one column df of the target for the next period


regressor.fit(X_train , y_train)

predicted_val = regressor.predict(X_test)

name="XGBM_test"
d=ErrorCalculator(name,y_test,predicted_val)
print(d)

#load the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
ConsumptionFilePath=os.path.join(OutputPath,Results_DF_FileName)

df_errors_cumulative = pd.read_csv(ConsumptionFilePath, index_col=0, parse_dates=[0])

#concat the new errors
df_errors = pd.DataFrame(data={k: [v] for k, v in d.items()})
df_errors = df_errors.set_index('Pipelines')
df_errors=pd.concat([df_errors_cumulative, df_errors], ignore_index=False)

#save the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
Joined_DF_FilePath=os.path.join(OutputPath,Results_DF_FileName)
df_errors.to_csv(Joined_DF_FilePath)

# %%
df_errors

# %%
predicted_DF = pd.DataFrame(predicted_val , index=DF_test.index, columns=["XGBM_test_AF"])

predicted_DF = predicted_DF.join(y_test)

predicted_DF=predicted_DF.loc[DF_test.index]

predicted_DF.columns = ["XGBM_test_AF", "NetLoad"]


#predicted_DF.plot()

# %% [markdown]
# # get the feature importance 

# %%
correlation_matrix = DF_training_and_DF_validation.corr(method="spearman")
# Identify columns with absolute correlation to "target" below 0.1
low_corr_columns = correlation_matrix.index[correlation_matrix[target].abs() < 0.1]

# Drop those columns from the DataFrame
df.drop(columns=low_corr_columns, inplace=True)

total_size = len(df)
# Step 3: Split the data
DF_training = df.iloc[:total_size - 2 * validation_test_size]
DF_validation = df.iloc[total_size - 2 * validation_test_size:total_size - validation_test_size]
DF_test = df.iloc[total_size - validation_test_size:]

DF_training_and_DF_validation=pd.concat([DF_training,DF_validation])

# Step 4: Verify the sizes of each set
print("Training set size:", len(DF_training))
print("Validation set size:", len(DF_validation))
print("Test set size:", len(DF_test))


# %%
def get_feature_importance(regressor,DF_training_and_DF_validation):


    DF_training_and_DF_validation=pd.concat([DF_training,DF_validation])

    X_train=DF_training_and_DF_validation.drop([target],axis=1) # df, all features except target in a dataframe for the first period
    y_train=DF_training_and_DF_validation[target] #one column df, of the target for the first period

    regressor.fit(X_train , y_train)
    # Extract feature importance using 'gain' (other options: 'weight', 'cover', 'total_gain', 'total_cover')
    importance_dict = regressor.get_booster().get_score(importance_type='total_gain')

    importance_series_total_gain = pd.Series(importance_dict).sort_values(ascending=False)

    importance_dict = regressor.get_booster().get_score(importance_type='total_cover')

    importance_series_total_cover = pd.Series(importance_dict).sort_values(ascending=False)

    # Step 1: Rank the features in both total_gain and total_cover series
    rank_total_gain = importance_series_total_gain.rank(ascending=False, method='average')
    rank_total_cover = importance_series_total_cover.rank(ascending=False, method='average')

    # Step 2: Combine the ranks into a DataFrame for easy manipulation
    ranks_df = pd.DataFrame({
        'Rank_Total_Gain': rank_total_gain,
        'Rank_Total_Cover': rank_total_cover
    })

    # Step 3: Compute the average rank for each feature
    ranks_df['Average_Rank'] = ranks_df.mean(axis=1)

    # Step 4: Sort by the average rank in ascending order (lower rank is better)
    final_ranked_features = ranks_df.sort_values(by='Average_Rank')

    return final_ranked_features

"""
If you are more concerned about how much a feature contributes to improving model accuracy, use total_gain because it shows how much a feature has helped in reducing prediction errors.
If you want to know how often a feature impacts a large number of samples, use total_cover, as it indicates how influential a feature is in terms of affecting many data points.
"""

# %%
final_ranked_features=get_feature_importance(regressor,DF_training_and_DF_validation) #output df
xgboost_ranked_features=final_ranked_features.index.to_list() #take the index and make it list

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# # Feature selection

# %%
def plot_errors (ErrorSeries):
  """
  This is a function that plots the features that are not
  """
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import matplotlib.path as mpath
  import numpy as np

  import matplotlib.pyplot as plt
  import numpy as np

  x = np.arange(len(ErrorSeries.index))
  y = ErrorSeries.values
  labels = ErrorSeries.index

  plt.figure(1,figsize=(13,5))
  plt.style.use("seaborn-v0_8-whitegrid")
  plt.plot(x, y)

  plt.xticks(x, labels, rotation =40)
  plt.ylabel('MSE [W^2]', wrap=True)
  plt.xlabel('Features', wrap=True)


  plt.margins(0.05)

  plt.subplots_adjust(bottom = 0.05)
  plt.show()
  

def FeatureSelectionWithSHAP1_fixed_test(regressor, DF_features, DF_target, ordered_features_list, test_size=672, tolerance=100):
    """
    This function receives a regressor model, the features that the model was trained with,
    and the target that it had to forecast. Starting from the most important feature, 
    we find the error of the TimeSeries Cross-Validation with a fixed test size.
    By adding features, we find the new error of the forecast.

    Parameters:
    - regressor: The regression model to use for training and prediction.
    - DF_features: DataFrame containing the features.
    - DF_target: Series containing the target variable.
    - ordered_features_list: List of features ordered by importance (e.g., from SHAP analysis).
    - test_size: Number of steps to use in the test set (default is 672).
    - tolerance: Number of features to add before stopping if no improvement in error (default is 20).

    Returns:
    - ErrorSeries: A pandas Series with the errors for each step of feature addition.
    """

    feature_list = []  # Empty list of features
    error_list = []  # Empty list to store errors for each set of features
    total_samples = len(DF_features)  # Total number of samples in the dataset
    n_splits = 5  # Number of splits (fixed)
    no_improvement_count = 0  # Count features added without improvement
    min_error = float('inf')  # Start with a large error to track the minimum error

    for i in ordered_features_list:
        # Start the loop with the best feature and append the next ones
        feature_list.append(i)

        X = DF_features[feature_list].to_numpy()
        y = DF_target.to_numpy()

        #print(f"Performing feature selection with features: {feature_list}")

        # Custom logic to create splits with a fixed test size of 672
        splits = []
        start_train_size = total_samples - (n_splits * test_size)  # Calculate where to start training

        for split in range(n_splits):
            train_end = start_train_size + split * test_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end <= total_samples:  # Ensure the test set is within the bounds
                splits.append((list(range(0, train_end)), list(range(test_start, test_end))))

        TimeSeriesCVerror = []  # MSE errors for each fold

        # Time series cross-validation with fixed test size
        for train_index, test_index in splits:
            #print(f"TRAIN: {train_index}, TEST: {test_index}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the regressor and predict
            regressor.fit(X_train, y_train)
            predicted_val = regressor.predict(X_test)
            
            # Calculate the error for this fold
            Error = mean_squared_error(y_test, predicted_val)
            TimeSeriesCVerror.append(Error)
            #print(f"This is the error for one TS iteration: {Error}")

        # Calculate the average error across all splits
        TS_CV_error = sum(TimeSeriesCVerror) / len(TimeSeriesCVerror)
        #print(f"Cumulative error of the last steps: {TS_CV_error}")
        error_list.append(TS_CV_error)  # Store the error for this set of features

        # Check if the error improved
        if TS_CV_error < min_error:
            min_error = TS_CV_error  # Update the minimum error
            no_improvement_count = 0  # Reset the no-improvement count
        else:
            no_improvement_count += 1  # Increment if there's no improvement

        # Break the loop if no improvement is observed after 20 features
        if no_improvement_count >= tolerance:
            print(f"No improvement after {tolerance} features. Stopping early.")
            break

    # Create a pandas Series to store the error associated with each feature
    ErrorSeries = pd.Series(error_list, index=feature_list)

    # Plot the errors using a custom plot function
    #plot_errors(ErrorSeries)

    return ErrorSeries

# %%
DF_training_and_DF_validation=pd.concat([DF_training,DF_validation])
DF_target = DF_training_and_DF_validation[target]
DF_features = DF_training_and_DF_validation.drop(target , axis=1)
DF_features=DF_features[xgboost_ranked_features]


# %%
#selecting only the features from the xgbm model
FirstSeries=FeatureSelectionWithSHAP1_fixed_test(regressor,DF_features,DF_target,xgboost_ranked_features)


# %%
def select_features_minimum_plus_others(series):
    """
    Takes a pandas Series and selects the features that:
    - Include all features up to the minimum error.
    - After the minimum error, only include features that reduce the error compared to the previous one.
    - Ensures no duplicate features are added.

    Parameters:
    - series: A pandas Series where index are feature names and values are errors.

    Returns:
    - A list of selected feature names without duplicates.
    """
    # Find the index of the minimum value
    min_idx = series.idxmin()

    # Select all features up to and including the minimum
    selected_features = list(dict.fromkeys(series[:min_idx].index.tolist() + [min_idx]))

    # After the minimum, keep only the features that decrease the error
    after_min_series = series[min_idx:]

    # Loop through the series after the minimum value and add features that decrease the error
    for i in range(1, len(after_min_series)):
        if after_min_series[i] < after_min_series[i - 1]:
            feature = after_min_series.index[i]
            if feature not in selected_features:
                selected_features.append(feature)

    return selected_features

def keep_indices_till_min(series):
    """
    Keeps all index values from the series up to and including the minimum value using a for loop,
    while ensuring no duplicates are added.

    Parameters:
    - series: A pandas Series where the index are feature names and the values are errors.

    Returns:
    - A list of unique index values (features) up to and including the minimum error value.
    """
    # Initialize an empty list to store the selected indices
    selected_features = []

    # Find the minimum value in the series
    min_value = series.min()

    # Loop over the series
    for idx, value in series.items():
        # Add the current index to the selected features only if it's not already present
        if idx not in selected_features:
            selected_features.append(idx)
        
        # If the current value is the minimum, stop the loop
        if value == min_value:
            break

    return selected_features



# %%
#features_after_FS1=select_features_minimum_plus_others(FirstSeries)
features_after_FS1=keep_indices_till_min(FirstSeries)
#features_after_FS1=FirstSeries[:50].index.to_list()

# %% [markdown]
# ## Checkpoint

# %%
#save the first series which is the error and the features
FSname1="selected_features_and_errors.csv"
Joined_DF_FilePath=os.path.join(OutputPath,FSname1)
FirstSeries.to_csv(Joined_DF_FilePath)

# %%
#save the features
FSname1 = "selected_features_step1.csv"
FilePath = os.path.join(OutputPath, FSname1)
features_after_FS1_df = pd.DataFrame(features_after_FS1, columns=["Features"])

# Save the DataFrame to the constructed file path
features_after_FS1_df.to_csv(FilePath, index=False)


# %%
# Read the CSV file into a DataFrame
FSname1 = "selected_features_step1.csv"
FilePath = os.path.join(OutputPath, FSname1)
features_after_FS1_df = pd.read_csv(FilePath)
features_after_FS1 = features_after_FS1_df["Features"].tolist()



# %% [markdown]
# # HPO feature selection

# %%
import numpy as np
from sklearn.model_selection import BaseCrossValidator

# Define a custom time series splitter
class CustomTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=3, test_size=96*7):
        self.n_splits = n_splits
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        for i in range(self.n_splits):
            test_end = n_samples - i * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start
            if train_end <= 0:
                break
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            yield train_indices, test_indices

# %%

def objective(trial):
    # Sample hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.1, 0.3)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
    
    # Define the regressor with sampled hyperparameters
    regressor = xgboost.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        tree_method='gpu_hist',     # Ensure GPU availability
        predictor='gpu_predictor',  # Ensure GPU availability
    )
    print("this is an iteration", flush=True)
    # Get feature importance
    final_ranked_features = get_feature_importance(regressor, DF_training_and_DF_validation)
    xgboost_ranked_features = final_ranked_features.index.to_list()
    
    # Prepare features and target
    DF_target = DF_training_and_DF_validation[target]
    DF_features = DF_training_and_DF_validation.drop(target, axis=1)
    DF_features = DF_features[xgboost_ranked_features]
    
    # Feature selection using SHAP
    FirstSeries = FeatureSelectionWithSHAP1_fixed_test(regressor, DF_features, DF_target, xgboost_ranked_features)
    features_after_FS1 = keep_indices_till_min(FirstSeries)
    
    # Update features based on selection
    DF_target = DF_training_and_DF_validation[target]
    DF_features = DF_training_and_DF_validation[features_after_FS1]
    
    
    # Initialize the custom time series splitter
    tscv = CustomTimeSeriesSplit(n_splits=5, test_size=96*7)
    errors = []

    # Perform cross-validation using the custom splitter
    for train_index, test_index in tscv.split(DF_features):
        # Use the same selected features for all folds
        X_train_fold = DF_features.iloc[train_index][features_after_FS1]
        X_test_fold = DF_features.iloc[test_index][features_after_FS1]
        y_train_fold = DF_target.iloc[train_index]
        y_test_fold = DF_target.iloc[test_index]

        # Fit the model
        regressor.fit(X_train_fold, y_train_fold)

        # Make predictions
        predicted_val = regressor.predict(X_test_fold)

        # Calculate error
        d = ErrorCalculator("XGBM_SF_val", y_test_fold, predicted_val)
        errors.append(d["MSE"])
    
    # Compute the average error across folds
    average_error = np.mean(errors)
    
    # Store the features used in this trial
    trial.set_user_attr('features', features_after_FS1)
    
    # Return the average MSE as the objective to minimize
    return average_error

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Access the best trial and its features
best_trial = study.best_trial
print(f"Best trial MSE: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")
print(f"Features used in best trial: {best_trial.user_attrs['features']}")

# %%
# Define file structure similar to Results_DF_FileName
"""Results_JSON_FileName = f"Optuna_Params_{scenario}_{season}_.json"
Joined_JSON_FilePath = os.path.join(OutputPath, Results_JSON_FileName)

# Save best trial parameters to JSON
with open(Joined_JSON_FilePath, "w") as f:
    json.dump(best_trial.params, f, indent=4)

print(f"Saved best trial parameters to {Joined_JSON_FilePath}")
"""
# %%
"""
# Define your data and target variables (make sure these are accessible)
# DF_training, DF_validation, DF_training_and_DF_validation, target

def objective(trial):
    # Sample hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.1, 0.3)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
    
    # Define the regressor with sampled hyperparameters
    regressor = xgboost.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        tree_method='gpu_hist',     # Ensure GPU availability
        predictor='gpu_predictor',  # Ensure GPU availability
    )
    
    # Get feature importance
    final_ranked_features = get_feature_importance(regressor, DF_training_and_DF_validation)
    xgboost_ranked_features = final_ranked_features.index.to_list()
    
    # Prepare features and target
    DF_target = DF_training_and_DF_validation[target]
    DF_features = DF_training_and_DF_validation.drop(target, axis=1)
    DF_features = DF_features[xgboost_ranked_features]
    
    # Feature selection using SHAP
    FirstSeries = FeatureSelectionWithSHAP1_fixed_test(regressor, DF_features, DF_target, xgboost_ranked_features)
    features_after_FS1 = keep_indices_till_min(FirstSeries)
    
    # Update features based on selection
    DF_target = DF_training_and_DF_validation[target]
    DF_features = DF_training_and_DF_validation[features_after_FS1]
    
    # Split data into training and validation sets
    X_train = DF_training[features_after_FS1]
    X_test = DF_validation[features_after_FS1]
    y_train = DF_training[target]
    y_test = DF_validation[target]
    
    # Fit the model
    regressor.fit(X_train, y_train)
    
    # Make predictions
    predicted_val = regressor.predict(X_test)
    
    # Calculate error
    name = "XGBM_SF_val"
    d = ErrorCalculator(name, y_test, predicted_val)
    
    # Store the features used in this trial
    features = X_train.columns.tolist()
    trial.set_user_attr('features', features)
    
    # Return the MSE as the objective to minimize
    return d["MSE"]

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Access the best trial and its features
best_trial = study.best_trial
print(f"Best trial MSE: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")
print(f"Features used in best trial: {best_trial.user_attrs['features']}")"""

# %%
#save the features
FSname1 = "selected_features_step1.csv"
FilePath = os.path.join(OutputPath, FSname1)
features_after_FS1=best_trial.user_attrs['features']

features_after_FS1_df = pd.DataFrame(features_after_FS1, columns=["Features"])

# Save the DataFrame to the constructed file path
features_after_FS1_df.to_csv(FilePath, index=False)

# %%
# Read the CSV file into a DataFrame
FSname1 = "selected_features_step1.csv"
FilePath = os.path.join(OutputPath, FSname1)
features_after_FS1_df = pd.read_csv(FilePath)
features_after_FS1 = features_after_FS1_df["Features"].tolist()

# %%
features_after_FS1

# %%
features_after_FS1=best_trial.user_attrs['features']
best_params=best_trial.params

regressor = xgboost.XGBRegressor(
        tree_method='gpu_hist',     # Ensure GPU availability
        predictor='gpu_predictor', # Ensure GPU availability
        **best_params 
    )

# %%
regressor

# %%
best_params

# %%
features_after_FS1

# %% [markdown]
# # Xgboost for validation with selected features

# %%
#features_after_FS1=Features_NextStep.index.to_list()

# %%
DF_target = DF_training_and_DF_validation[target]
DF_features = DF_training_and_DF_validation[features_after_FS1]

# %%
X_train=DF_training[features_after_FS1] # only selected features
X_test=DF_validation[features_after_FS1] # only selected
y_train=DF_training[target] #one column df, of the target for the first period
y_test=DF_validation[target] #one column df of the target for the next period

regressor.fit(X_train , y_train)

predicted_val = regressor.predict(X_test)

name="XGBM_SF_val"
d=ErrorCalculator(name,y_test,predicted_val)
d['features'] = len(X_train.columns)
print(d)

#load the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
ConsumptionFilePath=os.path.join(OutputPath,Results_DF_FileName)

df_errors_cumulative = pd.read_csv(ConsumptionFilePath, index_col=0, parse_dates=[0])

# %%
df_errors = pd.DataFrame(data={k: [v] for k, v in d.items()})

df_errors = df_errors.set_index('Pipelines')
df_errors=pd.concat([df_errors_cumulative, df_errors], ignore_index=False)

# %%
#save the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
Joined_DF_FilePath=os.path.join(OutputPath,Results_DF_FileName)
df_errors.to_csv(Joined_DF_FilePath)

# %%
df_errors

# %%
predicted_DF_split = pd.DataFrame(predicted_val , index=DF_validation.index, columns=["XGBM_val_SF"])

predicted_DF_split = predicted_DF_split.join(y_test)

predicted_DF_split=predicted_DF_split.loc[DF_validation.index]

predicted_DF_split.columns = ["XGBM_val_SF", "NetLoad"]


#predicted_DF_split.plot()

# %% [markdown]
# # Xgboost for test with selected features

# %%
DF_training_and_DF_validation=pd.concat([DF_training,DF_validation])

X_train=DF_training_and_DF_validation[features_after_FS1] # df, all features except target in a dataframe for the first period
X_test=DF_test[features_after_FS1] #df , all feature except target (same as before but for the next available period period)
y_train=DF_training_and_DF_validation[target] #one column df, of the target for the first period
y_test=DF_test[target] #one column df of the target for the next period


regressor.fit(X_train , y_train)

predicted_val = regressor.predict(X_test)

name="XGBM_SF_test"
d=ErrorCalculator(name,y_test,predicted_val)
print(d)

#load the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
ConsumptionFilePath=os.path.join(OutputPath,Results_DF_FileName)

df_errors_cumulative = pd.read_csv(ConsumptionFilePath, index_col=0, parse_dates=[0])

#concat the new errors
df_errors = pd.DataFrame(data={k: [v] for k, v in d.items()})
df_errors = df_errors.set_index('Pipelines')
df_errors=pd.concat([df_errors_cumulative, df_errors], ignore_index=False)

#save the errors
Results_DF_FileName="Error_Results_"+scenario+"_"+season+"_.csv"
Joined_DF_FilePath=os.path.join(OutputPath,Results_DF_FileName)
df_errors.to_csv(Joined_DF_FilePath)

# %%
df_errors

# %%
predicted_DF = pd.DataFrame(predicted_val , index=DF_test.index, columns=["XGBM_test_SF"])

predicted_DF = predicted_DF.join(y_test)

predicted_DF=predicted_DF.loc[DF_test.index]

predicted_DF.columns = ["XGBM_test_SF", "NetLoad"]


#predicted_DF.plot()


print(f"Best hyperparameters: {best_trial.params}")
