# Load library
import pickle
import pandas as pd
import numpy as np


# load dataset
df = pd.read_csv("data/youtube_ad_revenue_dataset.csv")
# print(f"dataset has duplicat values: {df.duplicated().any()}")

# remove duplicate values
df = df.drop_duplicates() 
# print(df.isnull().sum())

# fill null values
nullcalue_column = ["average_view_duration_sec","ctr_percent","subscribers_gained","cpm_usd"]
df[nullcalue_column] = df[nullcalue_column].fillna(df[nullcalue_column].mean())

# drop column"video_id" 
df = df.drop("video_id",axis = 1) 

# Log-transform skewed numeric features
skewed_cols = ['views', 'impressions', 'watch_time_hours', 'likes',
               'comments', 'shares', 'monetized_playbacks', 'subscribers_gained']

for col in skewed_cols:
    df[col] = np.log1p(df[col])

# Log-transform the target too
df['estimated_ad_revenue_usd_log'] = np.log1p(df['estimated_ad_revenue_usd'])

# onehot-encoding
df = pd.get_dummies(df,columns=["category","country"],drop_first=True)

# targat variable
Y = df.loc[:,'estimated_ad_revenue_usd_log']
X = df.drop(columns =["estimated_ad_revenue_usd","estimated_ad_revenue_usd_log"])

# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

# Model selection
print("Model: multiple linear regression")
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
reg_model_fit =regressor.fit(X_train,y_train)
# print(reg_model_fit.coef_)

# Train model prediction
y_train_pred_log = reg_model_fit.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)
y_train_actual = np.expm1(y_train)

# Test model prediction
y_pred_log = reg_model_fit.predict(X_test)
y_pred = np.expm1(y_pred_log)              # back to dollars
y_test_actual = np.expm1(y_test)    

# cheak accuracy
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae = mean_absolute_error(y_test_actual, y_pred)
print("MAE:", mae)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
Linearmodel_test_accuracy = r2_score(y_test_actual, y_pred)
print("Train R2 Score: ",r2_score(y_train_actual, y_train_pred))
print("Test R2 Score: ",Linearmodel_test_accuracy)

# ploat data
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,6))
# plt.scatter(y_test_actual, y_pred, alpha=0.7, label="Predictions")


# # best fit line
# min_val = min(y_test_actual.min(), y_pred.min())
# max_val = max(y_test_actual.max(), y_pred.max())
# plt.plot([min_val,max_val],[min_val,max_val],color = "red")  
# plt.title("Actual vs Predicted")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.show()


# model: grid search
# print("Model: Grid Search")
# from sklearn.model_selection import GridSearchCV
# param_grid  = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [10, 15, 20],
#     "min_samples_leaf": [5, 10, 15]
# }
# grid_search = GridSearchCV(
#                             RandomForestRegressor(random_state = 2),
#                             param_grid ,
#                             cv = 3,
#                             scoring = 'r2')
# grid_search.fit(X_train,y_train)
# print("Grid Search best parameters:", grid_search.best_params_)
# print("Best cv score:", grid_search.best_score_)


# model: random forest
print("Model: Random Forest")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 150, 
                           random_state = 2,
                           max_depth= 15,
                           min_samples_leaf = 10
                           )
rf.fit(X_train,y_train)

y_train_pred_log_rf = rf.predict(X_train)
y_test_pred_log_rf = rf.predict(X_test)

y_train_pred_rf = np.expm1(y_train_pred_log_rf)
y_test_pred_rf = np.expm1(y_test_pred_log_rf)


mae = mean_absolute_error(y_test_actual, y_test_pred_rf)
print("MAE:", mae)
mse = mean_squared_error(y_test_actual, y_test_pred_rf)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
RandomForest_test_accuracy = r2_score(y_test_actual, y_test_pred_rf)
print("Train R2 Score: ",r2_score(y_train_actual, y_train_pred_rf))
print("Test R2 Score: ",RandomForest_test_accuracy)

# XGboost
print("Model: XGboost")
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators = 500,
    max_depth = 6,
    learning_rate = 0.03,
    random_state = 2,
    n_jobs = -1,
    early_stopping_rounds=30
)
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train, test_size = 0.1, random_state = 2)
xgb.fit(
    X_tr,y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)


print("Best iteration:", xgb.best_iteration)

X_train_pred_log_xgb = xgb.predict(X_train)
X_test_pred_log_xgb = xgb.predict(X_test)

y_train_pred_xgb = np.expm1(X_train_pred_log_xgb)
y_test_pred_xgb = np.expm1(X_test_pred_log_xgb)


mae = mean_absolute_error(y_test_actual,y_test_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test_actual,y_test_pred_xgb))

print("MAE:", mae)
print("RMSE:", rmse)
XGboost_test_accuracy = r2_score(y_test_actual, y_test_pred_xgb)
print("Train R2 Score: ",r2_score(y_train_actual, y_train_pred_xgb))
print("Test R2 Score: ",XGboost_test_accuracy)

best_accuracy = max(Linearmodel_test_accuracy,
                 RandomForest_test_accuracy,
                 XGboost_test_accuracy)

#save best model
if best_accuracy == Linearmodel_test_accuracy:
    best_model = regressor
    best_model_name = "Linear Regression"
elif best_accuracy == RandomForest_test_accuracy:
    best_model = rf
    best_model_name = "Random Forest"
else:
    best_model = xgb
    best_model_name = "XGboost"

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest Model Saved Successfully")

print("\nBest Model:")
print(best_model_name)

print("\nBest Accuracy:")
print(best_accuracy)

# Save the model columns
with open("model_columns.pkl", "wb") as file:
    pickle.dump(list(X.columns), file)
print("Model columns saved successfully!")
