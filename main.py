import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def gen_response_curves(x_name, y_name, n=1000, channels=2, curve_type='log'):
    response_curves = []
    for c in range(channels):
        x = np.arange(100,10000, 100)
        ltv = np.random.normal(loc=15, scale=5)
        beta_0 = np.random.normal(loc=-1000, scale=100)
        beta_1 = np.random.normal(loc=100, scale=10)
        error = np.random.normal(loc=0, scale=30+x/400)
        if curve_type=='log':
            y = beta_1*np.log(x) + error
        elif curve_type=='linear':
            y = beta_0+beta_1*x + error

        y=y.astype(int)
        channel = f'channel {c}'
        curve = pd.DataFrame({x_name:x, y_name:y, 'channel':channel, 'ltv':ltv})
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

def plot_profit_curve(data, x="marketing_spend", y="profit", optimal_spend=None):
    sns.scatterplot(x=x,
                    y=y,
                    data=data)
    if optimal_spend is not None:
        plt.axvline(optimal_spend, 0,1, color='red', linestyle='--')
        plt.text(x=optimal_spend+0.05*(data[x].max() - data[x].min()), y=1.1*data[y].min(), s="Optimal Marketing Spend", weight="bold", color='red')
    plt.title('Profit vs Marketing Costs')
    plt.show()

def fit_log_response_curve(data, x, y):
    model = smf.ols(formula=f'{y} ~ np.log({x}) - 1', data=data).fit()
    return model

def channel_profit(ltv, c, b, x, y):
    q = b*np.log(c)
    return q*ltv - c

def grid_search_opt_profit(data, budget):

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
            ltv = data[data.channel==channels[cnt]].ltv.values[0]
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
    return results

if __name__=="__main__":

    # data generation
    x = 'marketing_spend'
    y = 'acquisitions'
    data = gen_response_curves(x_name=x, y_name=y)
    #plot_response_curve(data)

    # estimation
    models = {}    
    for c in data.channel.unique():
        #cdata = data[data.channel == c].copy()
        model=fit_log_response_curve(data[data.channel == c], x=x, y=y)
        data.loc[data.channel == c, f'fitted_{y}'] = model.predict(data[data.channel == c])
        models[c]=model

    # review estimation
    plot_response_curve(data)
    
    # grid search method
    budget = 10000
    results = grid_search_opt_profit(data, budget)
    print(results)
    for channel in data.channel.unique():
        optimal_spend = results[f'optimal spend|{channel}'].values[0]
        print(optimal_spend)
        plot_profit_curve(results, x=f"channel marketing spend|{channel}", y="total profit", optimal_spend=optimal_spend)