import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def gen_response_curves(n=1000, channels=4, curve_type='log'):
    response_curves = []
    for c in range(channels):
        x = np.arange(100,10000, 100)
        beta_0 = np.random.normal(loc=-1000, scale=100)
        beta_1 = np.random.normal(loc=100, scale=10)
        error = np.random.normal(loc=0, scale=30+x/400)
        if curve_type=='log':
            y = beta_1*np.log(x) + error
        elif curve_type=='linear':
            y = beta_0+beta_1*x + error

        y=y.astype(int)
        channel = f'channel {c}'
        curve = pd.DataFrame({'marketing_spend':x, 'acquisitions':y, 'channel':channel})
        response_curves.append(curve)

    response_curves = pd.concat(response_curves)
    return response_curves

def plot_response_curve(data, x="marketing_spend", y="acquisitions", 
                        hue='channel', y_hat="fitted_acquisitions"):
    sns.scatterplot(x=x,
                    y=y,
                    hue=hue,
                    data=data)
    sns.lineplot(x=x, y=y_hat, hue=hue, data=data, linestyle='--')
    plt.title('Acquisitions wrt Marketing Spend by channel')
    plt.show()

def plot_profit_curve(data, x="marketing_spend", y="profit", 
                        hue='channel'):
    sns.scatterplot(x=x,
                    y=y,
                    hue=hue,
                    data=data)
    plt.title('Profit vs Marketing Costs')
    plt.show()

def fit_log_response_curve(data, x, y):
    model = smf.ols(formula=f'{y} ~ np.log({x}) - 1', data=data).fit()
    #print(model.summary())
    return model

def channel_profit(ltv, c, b, x, y):
    q = b*np.log(c)
    return q*ltv - c

if __name__=="__main__":
    data = gen_response_curves()
    x = 'marketing_spend'
    y = 'acquisitions'
    #plot_response_curve(data)

    models = {}    
    for c in data.channel.unique():
        cdata = data[data.channel == c].copy()
        #print(cdata)
        model=fit_log_response_curve(cdata, x=x, y=y)
        cdata[f'fitted_{y}'] = model.predict(cdata)
        models[c]=model
        plot_response_curve(cdata)
    
    c = data.channel.unique()[0]
    ltv=10
    profit_data = {'channel':[],
                   'ltv':[],
                   'marketing_spend':[],
                   'profit':[]}

    for cost in np.arange(100, 2000, 100):

        b = models[c].params[f'np.log({x})']
        profit = channel_profit(ltv, cost, b, x, y)
        profit_data['channel'].append(c)
        profit_data['ltv'].append(ltv)
        profit_data['marketing_spend'].append(cost)
        profit_data['profit'].append(profit)

    profit_data = pd.DataFrame(profit_data)
    plot_profit_curve(profit_data, x="marketing_spend", y="profit", 
                        hue='channel')
    optimal_cost = ltv*b
    print(optimal_cost)