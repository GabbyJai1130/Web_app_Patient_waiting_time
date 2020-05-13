import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

if __name__ == '__main__':
    df = pd.read_pickle('/Users/USER/Documents/Python/Data Analysis_Practice_GJ/ED_waiting_time/df_processed.pkl')

    features = df.drop('waiting_time',axis = 1)
    y = np.log1p(df['waiting_time'])

    scaler = MinMaxScaler()
    features.iloc[:,:2] = scaler.fit_transform(features.iloc[:,:2])

    x = features
    y_scaled = scaler.fit_transform(y.values.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.3, random_state=42)

    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=100,
                                     learning_rate=0.05,
                                     reg_lambda=1.3)

    xgb_regressor.fit(X_train.values, y_train)
    y_pred = xgb_regressor.predict(X_test.values)

    y_test_unsc = scaler.inverse_transform(y_test)
    y_pred_unsc = scaler.inverse_transform(y_pred.reshape(-1, 1))

    y_test_orginal = np.expm1(y_test_unsc)
    y_pred_orginal = np.expm1(y_pred_unsc)

    RMSE = np.sqrt(mean_squared_error(y_test_orginal, y_pred_orginal))

    print("RMSE of prediction: %0.2f minutes" % (RMSE))

    pickle.dump(scaler,open('scaler.pkl','wb'))
    pickle.dump(xgb_regressor,
                open('xgb_model.pkl', 'wb'))
