from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from readline import redisplay
import requests 
import time
import pandas as pd
from typing import Union
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



#Chicago Community area names mapped to their number
COMMUNITY_AREA_MAP = {
     'rogers park': 1, 'west ridge': 2, 'uptown': 3, 'lincoln square': 4,
    'north center': 5, 'lake view': 6, 'lincoln park': 7, 'near north side': 8,
    'edison park': 9, 'norwood park': 10, 'jefferson park': 11, 'forest glen': 12,
    'north park': 13, 'albany park': 14, 'portage park': 15, 'irving park': 16,
    'dunning': 17, 'montclare': 18, 'belmont cragin': 19, 'hermosa': 20,
    'avondale': 21, 'logan square': 22, 'humboldt park': 23, 'west town': 24,
    'austin': 25, 'west garfield park': 26, 'east garfield park': 27,
    'near west side': 28, 'north lawndale': 29, 'south lawndale': 30,
    'lower west side': 31, 'loop': 32, 'the loop': 32, 'near south side': 33,
    'armour square': 34, 'douglas': 35, 'oakland': 36, 'fuller park': 37,
    'grand boulevard': 38, 'kenwood': 39, 'washington park': 40, 'hyde park': 41,
    'woodlawn': 42, 'south shore': 43, 'chatham': 44, 'avalon park': 45,
    'south chicago': 46, 'burnside': 47, 'calumet heights': 48, 'roseland': 49,
    'pullman': 50, 'south deering': 51, 'east side': 52, 'west pullman': 53,
    'riverdale': 54, 'hegewisch': 55, 'garfield ridge': 56, 'archer heights': 57,
    'brighton park': 58, 'mckinley park': 59, 'bridgeport': 60, 'new city': 61,
    'west elsdon': 62, 'gage park': 63, 'clearing': 64, 'west lawn': 65,
    'chicago lawn': 66, 'west englewood': 67, 'englewood': 68,
    'greater grand crossing': 69, 'ashburn': 70, 'auburn gresham': 71,
    'beverly': 72, 'washington heights': 73, 'mount greenwood': 74,
    'morgan park': 75, "ohare": 76, 'edgewater': 77
}


## Function to get the Chicago community area number for a given address
def get_chicago_community_area_number(address: str) -> Union[int, None]:
    
    #accesses a geocoding service that can convert addresses to geographic coordinates
    nominatim_url = "https://nominatim.openstreetmap.org/search"

    # Parameters for the geocoding request
    nominatim_params = {
        'q': f"{address}, Chicago, IL, USA",
        'format': 'json',
        'addressdetails': 1,
        'limit': 1
    }

    
    headers = {
        'User-Agent': 'Chicago Community Area Lookup Script' # Replace with your app's name
    }

    try:
        response = requests.get(nominatim_url, params=nominatim_params, headers=headers)
        response.raise_for_status()  
        data = response.json()

        if not data:
            print(f"Warning: Could not find a match for address: {address}")
            return None

        # mapping adress to comunity area 
        address_details = data[0].get('address', {})
        community_area_name = address_details.get('city_district') or address_details.get('suburb')

        # If the community area name is not found...
        if community_area_name:
            
            normalized_name = community_area_name.lower()
            community_area_number = COMMUNITY_AREA_MAP.get(normalized_name)

            if community_area_number:
                return community_area_number
            else:
                print(f"Warning: Could not map area name '{community_area_name}' to a number.")
                return None
        else:
            print(f"Warning: Could not determine community area for address: {address}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"An error occurred while parsing the API response: {e}")
        return None

## Dictionary to map service requests to categories
#Category mapping 
service_request_map = {
    # Streets & Sidewalks
    'Pothole in Street Complaint': 'Streets & Sidewalks',
    'Alley Pothole Complaint': 'Streets & Sidewalks',
    'Street Cleaning Request': 'Streets & Sidewalks',
    'Street Light Pole Damage Complaint': 'Streets & Sidewalks',
    'Street Light Pole Door Missing Complaint': 'Streets & Sidewalks',
    'Street Light Out Complaint': 'Streets & Sidewalks',
    'Alley Light Out Complaint': 'Streets & Sidewalks',
    'Traffic Signal Out Complaint': 'Streets & Sidewalks',
    'Sign Repair Request – All Other Signs': 'Streets & Sidewalks',
    'Sidewalk Inspection Request': 'Streets & Sidewalks',
    'Inspect Public Way Request': 'Streets & Sidewalks',

    # Waste & Sanitation
    'Missed Garbage Pick-Up Complaint': 'Waste & Sanitation',
    'Garbage Cart Maintenance': 'Waste & Sanitation',
    'Blue Recycling Cart': 'Waste & Sanitation',
    'Wire Basket Request': 'Waste & Sanitation',
    'Fly Dumping Complaint': 'Waste & Sanitation',
    'Sanitation Code Violation': 'Waste & Sanitation',
    'Clean Vacant Lot Request': 'Waste & Sanitation',
    'Weed Removal Request': 'Waste & Sanitation',

    # Water Services
    'Water On Street Complaint': 'Water Services',
    'Water in Basement Complaint': 'Water Services',
    'Low Water Pressure Complaint': 'Water Services',
    'No Water Complaint': 'Water Services',
    'Open Fire Hydrant Complaint': 'Water Services',
    'Water Lead Test Kit Request': 'Water Services',
    'Buildings – Plumbing Violation': 'Water Services',

    # Animal Control
    'Stray Animal Complaint': 'Animal Control',
    'Nuisance Animal Complaint': 'Animal Control',
    'Vicious Animal Complaint': 'Animal Control',
    'Report an Injured Animal': 'Animal Control',
    'Dead Animal Pick-Up Request': 'Animal Control',
    'Dead Bird': 'Animal Control',
    'Pet Wellness Check Request': 'Animal Control',

    # Trees & Yard Waste
    'Tree Emergency': 'Trees & Yard Waste',
    'Tree Debris Clean-Up Request': 'Trees & Yard Waste',
    'Tree Removal Inspection': 'Trees & Yard Waste',

    'Tree Planting Request': 'Trees & Yard Waste',
    'Yard Waste Pick-Up Request': 'Trees & Yard Waste',

    # Vehicle & Parking Services
    'Abandoned Vehicle Complaint': 'Vehicle & Parking Services',
    'Finance Parking Code Enforcement Review': 'Vehicle & Parking Services',
    'City Vehicle Sticker Violation': 'Vehicle & Parking Services',
    'Divvy Bike Parking Complaint': 'Vehicle & Parking Services',
    'E-Scooter Parking Complaint': 'Vehicle & Parking Services',
    'Bicycle Request/Complaint': 'Vehicle & Parking Services',
    'Cab Feedback': 'Vehicle & Parking Services',

    # Property & Building Code Enforcement
    'Building Violation': 'Property & Building Code Enforcement',
    'No Building Permit and Construction Violation': 'Property & Building Code Enforcement',
    'Graffiti Removal Request': 'Property & Building Code Enforcement',
    'Shared Housing/Vacation Rental Complaint': 'Property & Building Code Enforcement',
    'Restaurant Complaint': 'Property & Building Code Enforcement',
    'No Air Conditioning': 'Property & Building Code Enforcement',

    # Pest Control
    'Rodent Baiting/Rat Complaint': 'Pest Control',
    'Bee/Wasp Removal': 'Pest Control',

    # Miscellaneous & Other Services
    'Aircraft Noise Complaint': 'Miscellaneous & Other Services',
    'Wage Complaint': 'Miscellaneous & Other Services',
    '311 INFORMATION ONLY CALL': 'Miscellaneous & Other Services'
}





## Connecting to Chicago Request API 
api_url = 'https://data.cityofchicago.org/resource/v6vf-nfxy.csv'

all_data = []
limit = 1000
offset = 0
total_records_to_fetch = 100 * limit #(100 batches * 1000 records/batch)
records_fetched = 0

print("Starting data download...")

# Loop to download data in batches
while records_fetched < total_records_to_fetch:

    full_api_url = f"{api_url}?$limit={limit}&$offset={offset}"

    print(f"Fetching data from: {full_api_url}")

    try:

        chunk_df = pd.read_csv(full_api_url)


        if chunk_df.empty:
            print("No more data to fetch.")
            break


        all_data.append(chunk_df)
        records_fetched += len(chunk_df)



        offset += limit


        time.sleep(0.1)

    except Exception as e:
        print(f"Error fetching data chunk at offset {offset}: {e}")

        break


if all_data:
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nFinished downloading data. Total records downloaded: {len(df)}")

else:
    print("No data was downloaded.")
    df = pd.DataFrame()

# Remove rows where 'street_address' is NaN AND 'community_area' is also NaN
df = df[~(df['street_address'].isna() & df['community_area'].isna())]

#create a copy for the original dataframe
original_df = df.copy()

# Convert 'created_date' to datetime if not already
if not pd.api.types.is_datetime64_any_dtype(df['created_date']):
    # Convert 'created_date' to datetime if not already
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')

# Convert to MM-DD-YYYY string format
df['created_date'] = df['created_date'].dt.strftime('%m-%d-%Y')

# Remove rows with the latest date
latest_date_str = df['created_date'].max()
df = df[df['created_date'] != latest_date_str]

#save the dataframe to a csv file
df.to_csv("chicago_data.csv", index=False)
df = pd.read_csv('chicago_data.csv')

##Using the community area mapping, we link addresses to missing community areas

#First, we start with the df
missing_mask = pd.isna(df['community_area'])

print(f"Found {missing_mask.sum()} observations with missing community areas to process.\n")

for index in df[missing_mask].index:
        # Get the address from the current row
        address = df.loc[index, 'street_address']
        
        print(f"Processing address: '{address}' (Index: {index})")
        
        # Run the function to get the community area number
        community_area_num = get_chicago_community_area_number(address)
        
        # If a number was successfully found, update the DataFrame
        if community_area_num is not None:
            df.loc[index, 'community_area'] = community_area_num
            print(f"  -> Success! Found Community Area #{community_area_num}. Updated DataFrame.\n")
        else:
            print(f"  -> Failed. Could not determine community area.\n")

#Then, we do the orginal_df
missing_mask = pd.isna(original_df['community_area'])

print(f"Found {missing_mask.sum()} observations with missing community areas to process.\n")

for index in original_df[missing_mask].index:
        # Get the address from the current row
        address = original_df.loc[index, 'street_address']
        
        print(f"Processing address: '{address}' (Index: {index})")
        
        # Run the function to get the community area number
        community_area_num = get_chicago_community_area_number(address)
        
        # If a number was successfully found, update the DataFrame
        if community_area_num is not None:
            original_df.loc[index, 'community_area'] = community_area_num
            print(f"  -> Success! Found Community Area #{community_area_num}. Updated DataFrame.\n")
        else:
            print(f"  -> Failed. Could not determine community area.\n")

#remove rows with community areastill NaN
df = df[~df['community_area'].isna()]
original_df = original_df[~original_df['community_area'].isna()]

## merge with ASC community survey data
ACS_data_community = pd.read_csv('ACS_5_Year_Data_by_Community_Area_20250731.csv')


ACS_data_community['community_area_name_lower'] = ACS_data_community['Community Area'].str.lower()


ACS_data_community['community_area_number'] = ACS_data_community['community_area_name_lower'].map(COMMUNITY_AREA_MAP)


ACS_data_community.drop(columns = ['community_area_name_lower', 'Community Area'], inplace=True)

# Corrected column name from 'RecordID' to 'Record ID'
ACS_data_community = ACS_data_community.drop(columns=['Record ID'])

## Merge the two DataFrames on 'community_area_number'
merged_df = pd.merge(df, ACS_data_community, left_on='community_area', right_on='community_area_number', how='inner')

original_df_merged = pd.merge(original_df, ACS_data_community, left_on='community_area', right_on='community_area_number', how='inner')


##Adding temporal features and aggregating data

# Convert 'created_date' to datetime objects, coercing errors
merged_df['created_date'] = pd.to_datetime(merged_df['created_date'], errors='coerce')


# Assuming 'merged_df' is your DataFrame.

# Convert the 'created_date' column to datetime and remove timezone info
merged_df['created_date'] = pd.to_datetime(merged_df['created_date']).dt.tz_localize(None)

# Define the start of the time window (exactly 48 hours ago)
start_of_two_days_ago = (datetime.now() - timedelta(days=2))

##Create new dataframe that maps service requests to categories (will be helpful for dashboard)
merged_df['service_request_category'] = merged_df['sr_type'].map(service_request_map)
original_df_merged['service_request_category'] = original_df_merged['sr_type'].map(service_request_map)

# Filter for requests in the last 48 hours
filtered_df = merged_df[
    (merged_df['created_date'] >= start_of_two_days_ago) &
    (merged_df['created_date'] < datetime.now())
]


# Group by community area and service type and count the occurrences
service_counts = filtered_df.groupby(['community_area_number', 'service_request_category']).size().reset_index(name='count')

# Sort the counts to ensure the most frequent services come first for each area
sorted_counts = service_counts.sort_values(['community_area_number', 'count'], ascending=[True, False])

# Group by community area and aggregate the top 3 service types into a list
top_services_list = sorted_counts.groupby('community_area_number')['service_request_category'].apply(
    lambda x: x.head(3).tolist()
).reset_index(name='top_3_services')

#save the top services list to a CSV file
top_services_list.to_csv("top_services_list.csv", index=False)

# Extract features from 'created_date'
merged_df['created_date_formatted'] = merged_df['created_date'].dt.strftime('%m-%d-%y')
merged_df['created_year'] = merged_df['created_date'].dt.year
merged_df['created_day_of_week'] = merged_df['created_date'].dt.day_name()
merged_df['created_weekday_weekend'] = merged_df['created_date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Function to determine the season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

merged_df['created_season'] = merged_df['created_date'].dt.month.apply(get_season)

# Add 'is_holiday' feature
cal = calendar()
# Generate holidays for the range of years in your data
holidays = cal.holidays(start=merged_df['created_date'].min(), end=merged_df['created_date'].max())
merged_df['is_holiday'] = merged_df['created_date'].dt.normalize().isin(holidays)


# Group by 'community area', formatted date, and the new features and count the occurrences
community_area_counts = merged_df.groupby(['community_area', 'created_date_formatted', 'created_year', 'created_day_of_week',
                                     'created_weekday_weekend', 'created_season', 'is_holiday', 
                                     'Under $25,000', '$25,000 to $49,999', '$50,000 to $74,999', '$75,000 to $125,000', 
                                     '$125,000 +', 'Male 0 to 17', 'Male 18 to 24', 'Male 25 to 34', 'Male 35 to 49', 'Male 50 to 64', 
                                     'Male 65+', 'Female 0 to 17', 'Female 18 to 24', 'Female 25 to 34', 'Female 35 to 49',
                                       'Female 50 to 64', 'Female 65 +', 'Total Population', 'White', 'Black or African American',
                                         'American Indian or Alaska Native', 'Asian', 'Native Hawaiian or Pacific Islander', 
                                         'Other Race', 'Multiracial', 'White Not Hispanic or Latino', 'Hispanic or Latino'
                                     ]).size().reset_index(name='count')

# Convert 'created_date_formatted' to datetime for proper sorting and lagging
community_area_counts['created_date_formatted'] = pd.to_datetime(community_area_counts['created_date_formatted'], format='%m-%d-%y')

# Sort by ward and date to ensure correct lagging and rolling calculations
community_area_counts = community_area_counts.sort_values(by=['community_area', 'created_date_formatted'])

# Calculate lag variables
for i in range(1, 7):
    community_area_counts[f'count_lag_{i}'] = community_area_counts.groupby('community_area')['count'].shift(i)

# Calculate rolling window 7-day average
# Calculate rolling window 7-day average and assign it correctly
community_area_counts['count_rolling_avg_7'] = community_area_counts.groupby('community_area')['count'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate rolling window 7-day standard deviation for each community_area
community_area_counts['count_rolling_std_7'] = (
    community_area_counts
    .groupby('community_area')['count']
    .rolling(window=7, min_periods=1)
    .std()
    .reset_index(level=0, drop=True)
)


community_area_counts.to_csv("community_area_counts.csv", index=False)

# Remove all rows where any of the 'count_lag' columns have NaN values
lag_cols = [col for col in community_area_counts.columns if col.startswith('count_lag')]
community_area_counts_no_nan_lags = community_area_counts.dropna(subset=lag_cols)

## Preparing data for modeling

# Identify categorical columns to one-hot encode
# Exclude 'created_date_formatted' as it's a datetime object used for sorting/lagging
categorical_cols = ['created_day_of_week', 'created_weekday_weekend', 'created_season']

# Perform one-hot encoding
community_area_counts_encoded = pd.get_dummies(community_area_counts_no_nan_lags, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y) using the encoded DataFrame
features = [col for col in community_area_counts_encoded.columns if col != 'count']
X = community_area_counts_encoded[features]
y = community_area_counts_encoded['count']

# Separate numerical and categorical columns for scaling
numerical_cols = X.select_dtypes(include=np.number).columns
# Exclude date columns from scaling if they are still in X and not intended for scaling
# For example, if 'created_year' is in numerical_cols and you don't want to scale it:
numerical_cols = numerical_cols.drop('created_year', errors='ignore')


# Apply StandardScaler to numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# Apply log transformation to the target variable
y_log_transformed = np.log1p(y)

##XGBoost Model Training

# Assuming X, and y_log_transformed are already defined in previous cells

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)


# Define a reduced parameter grid for GridSearchCV to speed up execution
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],       
    'learning_rate': [0.01, 0.1, 0.15, 0.2],    
    'max_depth': [3, 5, 8, 12],            
    'colsample_bytree': [0.8, 0.85, 0.9],  
    'subsample': [0.8, 0.85, 0.9]          
}

# Initialize the XGBoost regressor model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Initialize GridSearchCV with the model, reduced parameter grid, and TimeSeriesSplit
# We use y_log_transformed for training as the target variable is on a log scale
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=tscv, verbose=1, n_jobs=-1)

# Fit GridSearchCV to the data (using X which includes numerical and one-hot encoded features)
# Note: GridSearchCV will handle the splitting internally using tscv
grid_search.fit(X.select_dtypes(include=np.number), y_log_transformed)


# Print the best parameters found by GridSearchCV
print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# You can now use best_model for making predictions
print("\nBest model obtained from GridSearchCV:")
print(best_model)

##Evaluation

predictions_train_log_transformed = best_model.predict(X.select_dtypes(include=['number']))

# Calculate the Mean Squared Error (MSE) on the log-transformed scale
mse_train_log_transformed = mean_squared_error(y_log_transformed, predictions_train_log_transformed)

# Calculate the Root Mean Squared Error (RMSE) on the log-transformed scale
rmse_train_log_transformed = np.sqrt(mse_train_log_transformed)

print(f"RMSE on the log-transformed training data using best_model: {rmse_train_log_transformed}")

# Inverse transform y_train to the original scale and display the head
y_train_original_scale = np.expm1(y_log_transformed)
print("\ny_train on original scale (head):")


predictions_train_original_scale = np.expm1(predictions_train_log_transformed)
mse_train_original_scale = mean_squared_error(y_train_original_scale, predictions_train_original_scale)
rmse_train_original_scale = np.sqrt(mse_train_original_scale)
print(f"RMSE on the original scale training data using best_model: {rmse_train_original_scale}")

## Making test predictions df

# Find the latest date in the dataset
latest_date = community_area_counts_no_nan_lags['created_date_formatted'].max()
print(latest_date)

# Filter for the latest date
latest_counts = community_area_counts_no_nan_lags[
    community_area_counts_no_nan_lags['created_date_formatted'] == latest_date
][['community_area', 'count', 'count_lag_1', 'count_lag_2', 'count_lag_3', 'count_lag_4', 'count_lag_5', 'count_lag_6', 'count_rolling_avg_7', 'count_rolling_std_7']]

test_latest_counts = latest_counts.copy()

test_latest_counts['count_lag_6'] = test_latest_counts['count_lag_5'].fillna(0)
test_latest_counts['count_lag_5'] = test_latest_counts['count_lag_4'].fillna(0)
test_latest_counts['count_lag_4'] = test_latest_counts['count_lag_3'].fillna(0)
test_latest_counts['count_lag_3'] = test_latest_counts['count_lag_2'].fillna(0)
test_latest_counts['count_lag_2'] = test_latest_counts['count_lag_1'].fillna(0)
test_latest_counts['count_lag_1'] = test_latest_counts['count'].fillna(0)


feature_cols = [
    'count', 'count_lag_1', 'count_lag_2', 'count_lag_3',
    'count_lag_4', 'count_lag_5', 'count_lag_6'
]

# Calculate the mean and std for each row by setting axis=1
test_latest_counts['count_rolling_avg_7'] = test_latest_counts[feature_cols].mean(axis=1)
test_latest_counts['count_rolling_std_7'] = test_latest_counts[feature_cols].std(axis=1)

test_latest_counts.drop(columns=['count'], inplace=True)

# Get the current date
current_date = datetime.now().strftime('%m-%d-%Y')

# Create a new DataFrame for the current community area with the current date
test_data = pd.DataFrame(columns=['community_area', 'created_date_formatted', 'created_year',
       'is_holiday', 'count_lag_1',
       'count_lag_2', 'count_lag_3', 'count_lag_4', 'count_lag_5',
       'count_lag_6', 'count_rolling_avg_7', 'count_rolling_std_7',
       'created_day_of_week_Monday', 'created_day_of_week_Saturday',
       'created_day_of_week_Sunday', 'created_day_of_week_Thursday',
       'created_day_of_week_Tuesday', 'created_day_of_week_Wednesday',
       'created_weekday_weekend_Weekend'])

# Ensure 'created_date' is in datetime format
original_df_merged['created_date'] = pd.to_datetime(original_df_merged['created_date'], errors='coerce')

# Find the most recent date in the dataset
latest_date = original_df_merged['created_date'].dt.date.max()

# Filter the DataFrame to get only the data from the latest day
latest_day_data = original_df_merged[original_df_merged['created_date'].dt.date == latest_date]



test_latest_counts['created_date_formatted'] = current_date

test_latest_counts['created_year'] = datetime.now().year

test_latest_counts['is_holiday'] = True if datetime.now().date() in holidays else False

test_latest_counts['created_year'] = pd.to_datetime(test_latest_counts['created_date_formatted'], errors='coerce').dt.year

test_latest_counts['created_weekday_weekend'] = pd.to_datetime(test_latest_counts['created_date_formatted'], errors='coerce').dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

test_latest_counts['created_day_of_the_week'] = pd.to_datetime(test_latest_counts['created_date_formatted'], errors='coerce').dt.day_name()



test_data_merged = pd.merge(test_latest_counts, ACS_data_community, left_on='community_area', right_on='community_area_number', how='inner')



test_data_merged.drop(columns=['community_area_number'], inplace=True)


columns_to_clean = [
    'Under $25,000', '$25,000 to $49,999', '$50,000 to $74,999',
    '$75,000 to $125,000', '$125,000 +', 'Male 0 to 17', 'Male 18 to 24',
    'Male 25 to 34', 'Male 35 to 49', 'Male 50 to 64', 'Male 65+',
    'Female 0 to 17', 'Female 18 to 24', 'Female 25 to 34',
    'Female 35 to 49', 'Female 50 to 64', 'Female 65 +',
    'Total Population', 'White', 'Black or African American',
    'American Indian or Alaska Native', 'Asian', 'Other Race', 'Multiracial',
    'White Not Hispanic or Latino', 'Hispanic or Latino'
]

# Loop through each column to clean and convert it
for col in columns_to_clean:
    # Check if the column exists in the DataFrame to avoid errors
    if col in test_data_merged.columns:
        # Step 1: Remove any characters that are not digits or a decimal point.
        # This handles '$', ',', '+', and other symbols.
        test_data_merged[col] = test_data_merged[col].astype(str).str.replace(r'[^\d.]', '', regex=True)

        # Step 2: Convert the cleaned column to a numeric type.
        # 'errors='coerce'' will turn any values that still can't be converted into NaN (Not a Number).
        test_data_merged[col] = pd.to_numeric(test_data_merged[col], errors='coerce')

        # Step 3: Fill any resulting NaN values with 0. This is a safe default for prediction.
        test_data_merged[col] = test_data_merged[col].fillna(0)

## Running the predictions


# One-hot encode the categorical columns in the test data
# This converts columns with text into numerical 0s and 1s
X_test_encoded = pd.get_dummies(test_data_merged, dummy_na=False)



# Align the columns of the test set with the training set
X_train_encoded_columns = best_model.get_booster().feature_names
X_test_aligned = X_test_encoded.reindex(columns=X_train_encoded_columns, fill_value=0)

# Identify numerical columns that still exist after encoding
numerical_cols_to_scale = [col for col in X.select_dtypes(include=np.number).columns if col in X_test_aligned.columns]
if 'created_year' in numerical_cols_to_scale:
    numerical_cols_to_scale.remove('created_year') # Exclude year from scaling

X_test_aligned[numerical_cols_to_scale] = scaler.transform(X_test_aligned[numerical_cols_to_scale])

# Make predictions 
pred_log = best_model.predict(X_test_aligned)
predictions = np.expm1(pred_log)

# Save predictions to the original DataFrame
test_data_merged['count_prediction'] = predictions.round(0).astype(int)

print(test_data_merged[['community_area', 'count_prediction']].head())

final_predictions_df = test_data_merged[['community_area', 'count_prediction']]

test_latest_counts_graphed = test_latest_counts.copy()

test_latest_counts_graphed = test_latest_counts_graphed[['community_area', 'count_lag_1', 'count_lag_2', 'count_lag_3']]
test_latest_counts_graphed

test_latest_counts_graphed_with_predictions = test_latest_counts_graphed.merge(final_predictions_df, on='community_area')

test_latest_counts_graphed_with_predictions.to_csv("test_latest_counts_graphed_with_predictions.csv", index=False)


