import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import helpers as hlp

import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms

if __name__ == '__main__':
    # 1. fetch or load historical data for the ticker
    data_historical = hlp.get_ticker_historical_data("GME", use_persisted_data=True)

    # 2. calculate all specified statistics
    data_statistics = hlp.add_statistics(data_historical,
                                         use_sign=True,
                                         use_move=True,
                                         use_range=True,
                                         sma_windows=(5, 15, 20),
                                         ema_alphas=(0.5, 0.25))

    # 3. visualize the data and statistics to help determine relevant features
    data_statistics.plot.line(figsize=(20, 12), subplots=True, stacked=False, layout=(-1, 3))
    plt.show()

    # 4. process the explanatory and signal data
    # choose relevant features and print correlation matrix
    feature_names = ["Volume", "Return", "Move", "Range", "SMA_5", "SMA_15", "EMA_0.25"]
    print(data_statistics[feature_names].corr())
    X_train, X_test, y_train, y_test = hlp.prepare_regression_data(data_statistics, feature_names)

    # 5. fit logistic regression models with optimal alpha parameter determined on grid
    # the regularized linear regression model have additional penalty for coefficient size

    # lasso regression cost  = least square cost + alpha*|coefficients|_l1
    logistic_regression_lasso = skl_lm.LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    # ridge regression cost  = least square cost + alpha*|coefficients|^2_l2
    logistic_regression_ridge = skl_lm.LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)

    #  set up a grid of parameters to find best one according to NMAE. here C= 1/alpha
    grid = dict()
    grid['C'] = np.arange(0.5, 5, 0.01)

    # define grid search classes
    search_lasso = skl_ms.GridSearchCV(logistic_regression_lasso, grid, scoring="neg_mean_absolute_error", n_jobs=-1)
    search_ridge = skl_ms.GridSearchCV(logistic_regression_ridge, grid, scoring="neg_mean_absolute_error", n_jobs=-1)

    # perform the search on defined parameter grid
    results_lasso = search_lasso.fit(X_train, y_train)
    results_ridge = search_ridge.fit(X_train, y_train)
    print(
        f"""Best negative MAE LASSO:  {results_lasso.best_score_}. Best C LASSO: {results_lasso.best_params_}. Coefficients LASSO: {results_lasso.best_estimator_.coef_}""")
    print(
        f"""Best negative MAE Ridge:  {results_ridge.best_score_}. Best C Ridge: {results_ridge.best_params_}. Coefficients Ridge: {results_ridge.best_estimator_.coef_}""")

    # run prediction test data and check accuracy
    y_predict_lasso = search_lasso.best_estimator_.predict(X_test)
    y_predict_ridge = search_ridge.best_estimator_.predict(X_test)
    print(f"""Predict accuracy best LASSO: {skl.metrics.accuracy_score(y_test, y_predict_lasso)}""")
    print(f"""Predict accuracy best Ridge: {skl.metrics.accuracy_score(y_test, y_predict_ridge)}""")
