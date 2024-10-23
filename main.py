import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def gen_response_curves(n=1000, channels=2, curve_type='log'):
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

def plot_profit_curve(data, x="marketing_spend", y="profit"):
    sns.scatterplot(x=x,
                    y=y,
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
        #plot_response_curve(cdata)
    
    ltv=15
    budget = 10000
    results = {
        'iteration':[],
        'channel':[],
        'channel marketing spend':[],
        'ltv':[],
        'b':[],
        'channel profit':[]
        }

    for i in range(100):
        w = np.random.rand(4)
        total = w.sum()
        w /= total
        cost=budget*w
        
        channels = data.channel.unique()
        total_profit=0
        for cnt in range(len(channels)):
            results['iteration'].append(i)
            results['channel'].append(channels[cnt])
            results['channel marketing spend'].append(cost[cnt])
            b = models[channels[cnt]].params[f'np.log({x})']
            profit = channel_profit(ltv, cost[cnt], b, x, y)
            results['ltv'].append(ltv)
            results['b'].append(b)
            results['channel profit'].append(profit)

    results = pd.DataFrame(results)
    results = pd.pivot_table(results, index='iteration', columns=['channel'],
                   values=['channel marketing spend', 'ltv', 'b', 'channel profit'])
    results.columns = results.columns.map('|'.join).str.strip('|')

    profit_cols = [col for col in results.columns if col.split('|')[0]=='channel profit']
    results['total profit'] = results[profit_cols]. sum(axis=1)
    for channel in channels:
        results[f'optimal spend|{channel}'] = results[f'b|{channel}']*results[f'ltv|{channel}']
    print(results)
    for channel in channels:
        plot_profit_curve(results, x=f"channel marketing spend|{channel}", y="total profit")