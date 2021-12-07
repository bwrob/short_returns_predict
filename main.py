import sklearn as skl
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
from sklearn.preprocessing import StandardScaler
import numpy as np

import helpers as hlp

if __name__ == '__main__':

    # 1. fetch or load historical data for the ticker
    data_historical = hlp.get_ticker_historical_data("GME", use_persisted_data=True)

    # 2. calculate all specified statistics
    data_statistics = hlp.add_statistics(data_historical, sma_windows=[5, 30, 100], ema_alphas=[0.5, 0.25])

    # 3. visualize the data and statistics
    # statistics_data.plot.line(figsize=(20, 12), subplots=True, stacked=False, layout=(-1, 3))
    # plt.show()

    # 4. process the explanatory and signal data
    # choose relevant features and normalize the data
    feature_names = ["Volume", "Sign", "Move", "Range", "SMA_5", "SMA_30", "SMA_100", "EMA_0.25"]
    X = data_statistics[feature_names]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # categorize signal data
    y_categorized = hlp.categorize_return_data(data_statistics, 0.0, 0.0)
    y_categorized = y_categorized
    # split data randomly to train and test population
    X_train, X_test, y_train, y_test = skl_ms.train_test_split(X_scaled, y_categorized, test_size=0.25, random_state=1)

    # 6.
    logistic_regression_model = skl_lm.LogisticRegression(penalty="l1", solver="liblinear")
    #logistic_regression_model.fit(X_train, y_train)
    #y_prediction = logistic_regression_model.predict(X_test)
    #print("Accuracy:", skl.metrics.accuracy_score(y_test, y_prediction))
    #print("Precision:", skl.metrics.precision_score(y_test, y_prediction))
    #print("Recall:", skl.metrics.recall_score(y_test, y_prediction))

    grid = dict()
    grid['C'] = np.arange(0, 1, 0.01)
    # define search
    search = skl_ms.GridSearchCV(logistic_regression_model, grid, scoring='neg_mean_absolute_error', n_jobs=-1)
    # perform the search
    results = search.fit(X_train, y_train)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
