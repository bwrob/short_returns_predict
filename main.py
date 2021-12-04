import helpers as hlp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Workflow of the main script:
    # 1. fetch or load historical data for the ticker
    # 2. calculate specified statistics
    # 3. visualize the data
    # 4. perform normalization of predictor variables
    # 5. fit scikit logistic regression models

    # 1.
    historical_data = hlp.get_ticker_historical_data("GME", use_persisted_data=True)
    # 2.
    statistics_data = hlp.add_statistics(historical_data, sma_windows=[5, 30, 100], ema_alphas=[0.5])

    # 3.
    statistics_data.plot.area(figsize=(12, 2), subplots=True, stacked=False, layout=(-1, 2))
    plt.show()
