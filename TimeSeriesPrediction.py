# Classes for building a model
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TimeSeriesModelling():
    from sklearn import metrics
    def __init__(self, df):
        self.df = df
        
    def set_date(self, date_index):
        self.df[date_index] = pd.to_datetime(self.df[date_index])
        self.df[date_index] = [x.date() for x in self.df[date_index]]
        self.df.set_index(date_index, inplace=True)

        
    def scale_data(self, df, scaler, values_index):
        self.scaled = scaler.transform(df[values_index].values.reshape(-1,1))
        return self.scaled

    
    def preprocessing(self, differenced, lagshift=1):
        self.diff = differenced
        for s in range(1,lagshift+1):
            self.diff['shift_{}'.format(s)] = differenced['sales'].shift(s)
        self.diff.dropna(inplace=True)
        array_fit = (self.diff.values)
        self.X, self.y = array_fit[:, 0:-1], array_fit[:, -1]
        self.X = self.X.reshape(self.X.shape[0], 1, self.X.shape[1])
        return self.X, self.y    


    def add_additional_feat(self, df, list_feat, scaler):
        full_extra = df.reset_index(drop=True)
        full_extra = full_extra.loc[(len(df) - len(self.X)):, list_feat]
        full_extra.reset_index(inplace=True, drop=True)
        #full_extra = scaler.transform(full_extra)
        self.X = pd.DataFrame(self.X.reshape(self.X.shape[0], self.X.shape[2]),
                              columns=['sales'.format(x) for x in range(self.X.shape[2])]).join(pd.DataFrame(full_extra))
        self.X = self.X.values.reshape(self.X.shape[0], 1, self.X.shape[1])
        return self.X

    
    def train_test_split(self, X, y, train_size=0.75):
        self.X_train = X[:round((len(X)*train_size))]
        self.X_test = X[round((len(X)*train_size)):]
        self.y_train = y[:round((len(y)*train_size))]
        self.y_test = y[round((len(y)*train_size)):]
        return self.X_train, self.X_test, self.y_train, self.y_test

    
    def plot_timeseries(self, values_index, train=False, test=False):
        self.df[values_index].plot(kind='line', figsize=(15,5))  
        if train == True:
            f, ax = plt.subplots(figsize=(15,5))
            plt.plot(self.X_train[:,:,0].reshape(-1,1))
            plt.show()
        if test == True:
            f, ax = plt.subplots(figsize=(15,5))
            plt.plot(self.X_test[:,:,0].reshape(-1,1))
            plt.show()




class BaselineModel:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        
    def create_baseline_model(self, plot=True):
        history = [x for x in self.X_train[:,:,0].reshape(-1,1)]
        predictions = list()
        for i in range(len(self.X_test[:,:,0].reshape(-1,1))):
            # make prediction
            predictions.append(history[-1])
            # observation
            history.append(self.X_test[:,:,0].reshape(-1,1)[i])
        # report performance
        rmse = np.sqrt(metrics.mean_squared_error(self.X_test[:,:,0].reshape(-1,1), predictions))
        mse = metrics.mean_squared_error(self.X_test[:,:,0].reshape(-1,1), predictions)
        print('RMSE: %.3f' % rmse)
        print('MSE: %.3f' % mse)
        self.baseline = predictions
        # line plot of observed vs predicted
        if plot == True:
            plt.subplots(figsize=(18,8))
            plt.plot(self.X_test[:,:,0].reshape(-1,1))
            plt.plot(predictions)
            plt.show()




class ModelEvaluation:
    def __init__(self, model):
        self.model = model
        
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        self.mse_test = self.model.evaluate(X_test, y_test, batch_size=1)
        self.mse_train = self.model.evaluate(X_train, y_train, batch_size=1)
        print('The model has a test MSE of {} and a train MSE of {}.'.format(self.mse_test, self.mse_train))
        
        self.y_pred = self.model.predict(X_test, batch_size=1)
        self.y_hat = self.model.predict(X_train, batch_size=1)
        
        self.r2_test = metrics.r2_score(y_test, self.y_pred)
        self.r2_train = metrics.r2_score(y_train, self.y_hat)
        
        print('The model has a test R2 of {} and a train R2 of {}.'.format(self.r2_test, self.r2_train))
        
    def plot_evaluation(self, y_train, y_test):
        plt.subplots(figsize=(15,8))
        plt.plot(y_train, c='darkorange')
        plt.plot(self.y_hat, c='teal')
        plt.title('Train dataset and predictions')
        plt.show()
        plt.subplots(figsize=(15,8))
        plt.plot(y_test, c='tomato')
        plt.plot(self.y_pred, c='indigo')
        plt.title('Test dataset and predictions')
        plt.show()
            
    
