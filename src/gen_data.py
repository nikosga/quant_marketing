import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def gen_response_curves(n=1000, channels=4, curve_type='log'):
    response_curves = []

    for c in range(channels):
        x = np.arange(100,10000, 100)
        beta = np.random.normal(loc=100, scale=10)
        error = np.random.normal(loc=0, scale=30+x/400)
        y = beta*np.log(x) + error
        y=y.astype(int)
        channel = f'channel {c}'
        curve = pd.DataFrame({'marketing_spend':x, 'acquisitions':y, 'channel':channel})
        response_curves.append(curve)

    response_curves = pd.concat(response_curves)
    return response_curves

def plot_response_curve(data):
    sns.scatterplot(x="marketing_spend",
                    y="acquisitions",
                    hue='channel',
                    data=data)
    plt.title('Acquisitions wrt Marketing Spend by channel')
    plt.show()

if __name__=="__main__":
    data = gen_response_curves()
    print(data)
    plot_response_curve(data)