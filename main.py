import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import cvxpy as cp
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
                        hue='channel', y_hat="fitted_acquisitions", save=False):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x,
                    y=y,
                    hue=hue,
                    data=data)
    sns.lineplot(x=x, y=y_hat, hue=hue, data=data, linestyle='--')
    plt.title('Acquisitions wrt Marketing Spend by channel')
    if save:
        plt.savefig('figures/response_curve.png', bbox_inches='tight')
    plt.show()

def plot_profit_curve(data, x="marketing_spend", y="profit", optimal_spend=None, save=False):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x,
                    y=y,
                    data=data)
    if optimal_spend is not None:
        plt.axvline(optimal_spend, 0,1, color='red', linestyle='--')
        plt.text(x=optimal_spend+0.05*(data[x].max() - data[x].min()), y=1.1*data[y].min(), s="Optimal Spend - Simulation", weight="bold", color='red')
    plt.title('Profit vs Marketing Costs')
    if save:
        plt.savefig('figures/profit_plot.png', bbox_inches='tight')
    plt.show()

def plot_frontier(data, channel_a, channel_b, max_channel_a, max_channel_b, save=False):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=channel_a,
                    y=channel_b,
                    data=data,
                    size='total profit',
                    sizes=(10, 100))

    plt.scatter(x=max_channel_a, y=max_channel_b, color='r', s=100)
    plt.title('Frontier Plot')
    plt.tight_layout()
    if save:
        plt.savefig('figures/frontier.png', bbox_inches='tight')
    plt.show()


def fit_log_response_curve(data, x, y):
    model = smf.ols(formula=f'{y} ~ np.log({x}) - 1', data=data).fit()
    return model

def channel_profit(ltv, c, b, x, y):
    q = b*np.log(c)
    return q*ltv - c

def sim_opt_profit(data, budget, models, x, y, iterations=5000):

    results = {
        'iteration':[],
        'channel':[],
        'channel marketing spend':[],
        'ltv':[],
        'b':[],
        'channel profit':[]
        }

    for i in range(iterations):
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

    maxi = results['total profit'].idxmax()
    
    report = {}
    for channel in channels:
        report[f'sim optimal spend|{channel}'] = round(results.loc[maxi, f'channel marketing spend|{channel}'],2)

    report[f'sim max profit'] = round(results.loc[maxi, 'total profit'], 2)
    return results, report

def nonlinear_programming_opt(budget, ltv_0, ltv_1, b_0, b_1):

    c_0   = cp.Variable(pos=True)
    c_1 = cp.Variable(pos=True)

    # Constraint
    constraint = [c_0 + c_1 <= budget]

    # Objective
    obj = cp.Maximize(b_0*cp.log(c_0)*ltv_0 - c_0 + b_1*cp.log(c_1)*ltv_1 - c_1)

    problem = cp.Problem(obj, constraint)
    problem.solve(verbose=False)
    return {'c_0':round(c_0.value, 2), 'c_1':round(c_1.value, 2), 'max profit':round(problem.value, 2)}

def lagrangian_opt(ltv_0, ltv_1, b_0, b_1, budget):

    c_0 = round(budget/(1 + (ltv_1*b_1) / (ltv_0*b_0)), 2)
    c_1 = round(budget - c_0, 2)
    max_profit = round(b_0*np.log(c_0)*ltv_0 - c_0 + b_1*np.log(c_1)*ltv_1 - c_1, 2)

    return {
        "channel marketing spend|channel 0":c_0,
        "channel marketing spend|channel 1":c_1,
        "max profit":max_profit
    }

if __name__=="__main__":

    # data generation
    x = 'marketing_spend'
    y = 'acquisitions'
    data = gen_response_curves(x_name=x, y_name=y)
    channels = data.channel.unique()
    # estimation
    models = {}    
    for c in channels:
        model=fit_log_response_curve(data[data.channel == c], x=x, y=y)
        data.loc[data.channel == c, f'fitted_{y}'] = model.predict(data[data.channel == c])
        models[c]=model

    # review estimation
    plot_response_curve(data, save=True)
    budget = 1000
    
    ltv_0 = data[data.channel == channels[0]].ltv.values[0]
    ltv_1 = data[data.channel == channels[1]].ltv.values[0]
    b_0 = models[channels[0]].params[f'np.log({x})']
    b_1 = models[channels[1]].params[f'np.log({x})']

    # lagrangian method
    results = lagrangian_opt(ltv_0, ltv_1, b_0, b_1, budget)
    print('lagrangian optimal spend', results)
    
    # nonlinear programming method
    results = nonlinear_programming_opt(budget, ltv_0, ltv_1, b_0, b_1)
    print('nonlinear opt', results)

    # simulation method
    results, report = sim_opt_profit(data, budget, models, x, y)
    plot_frontier(results, 
                  f"channel marketing spend|{channels[0]}", 
                  f"channel marketing spend|{channels[1]}",
                  report[f'sim optimal spend|{channels[0]}'],
                  report[f'sim optimal spend|{channels[1]}'], save=True
                  )
    for channel in channels:
        optimal_spend = report[f'sim optimal spend|{channel}']
        plot_profit_curve(results, x=f"channel marketing spend|{channel}", y="total profit", 
                          optimal_spend=optimal_spend, save=True)

    print(report)