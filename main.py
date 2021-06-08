import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pca_project(X, k):
    X = X.T
    mean = X.mean(axis=1)
    X_centered = X - mean[:, np.newaxis]
    u, s, vh = np.linalg.svd(X_centered, full_matrices=False)
    PPTX = (u[:, :k].T @ X_centered).T
    return -1 * PPTX


def load_shares():
    df = pd.read_csv('prices.csv')
    df_sec = pd.read_csv('securities.csv')
    df.head(5)
    print(df)
    mask = df['date'].apply(lambda x: x[:4] == '2016')
    df = df[mask]
    df['date'] = df['date'].str.replace(' 00:00:00', '')
    dates = set(df['date'])
    Traded_all_year_brands = list(
        df['symbol'].value_counts().reset_index(name="count").query('count >= {}'.format(len(dates)))["index"])
    df_GIC = df_sec.loc[df_sec["Ticker symbol"].isin(Traded_all_year_brands)][['Ticker symbol', 'GICS Sector']]
    brands_prices = df.loc[df["symbol"].isin(Traded_all_year_brands)][['symbol', 'close']].groupby('symbol')[
        'close'].apply(list)
    prices_DF = pd.DataFrame({'Ticker symbol': brands_prices.index, 'prices': brands_prices.values})
    df_to_return = df_GIC.set_index('Ticker symbol').join(prices_DF.set_index('Ticker symbol')).reset_index()
    prices = np.array(df_to_return["prices"].values.tolist())
    return list(df_to_return["Ticker symbol"]), prices, list(df_to_return["GICS Sector"])


def plot_sectors(proj, sectors, sectors_to_plot):
    Xvals = []
    Yvals = []
    for category in sectors_to_plot:
        for i, sector in enumerate(sectors):
            if sector == category:
                Xvals.append(proj[i][0])
                Yvals.append(proj[i][1])
        plt.scatter(Xvals, Yvals)
        Xvals = []
        Yvals = []
    plt.legend(sectors_to_plot)
    plt.show()


def main():
    df = pd.read_csv('prices.csv')
    df.head(5)
    print(df)
    mask = df['date'].apply(lambda x: x[:4] == '2016')
    df = df[mask]
    df = df[df['symbol'] == 'AAPL'].reset_index()
    apple_close_prices = df.close
    apple_close_prices.plot()
    # plt.show()
    symbols, prices, sectors = load_shares()
    proj = pca_project(prices, 2)
    plot_sectors(proj, sectors, ['Energy', 'Information Technology'])

    transformed_data = np.zeros([prices.shape[0], prices.shape[1] - 1])
    for i in range(prices.shape[1] - 1, 0, -1):
        transformed_data[:, i - 1] = np.log(prices[:, i]) - np.log(prices[:, i - 1])
    proj = pca_project(transformed_data, 2)
    plot_sectors(proj, sectors, ['Energy', 'Information Technology'])
    plot_sectors(proj, sectors, ['Financials', 'Information Technology'])
    plot_sectors(proj, sectors, ['Energy', 'Information Technology', 'Real Estate'])
    plot_sectors(proj, sectors, set(sectors))
    max_stock = symbols[np.argmax(proj[:, 1])]
    plt.plot(prices[np.argmax(proj[:, 1])])
    plt.plot(prices[15])
    plt.legend([max_stock, symbols[15]])
    plt.show()


if __name__ == "__main__":
    main()
