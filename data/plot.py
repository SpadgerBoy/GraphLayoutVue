import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from matplotlib.ticker import MultipleLocator

def smooth(data, weight=0.95):
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    # smoothed[0] = data[0]
    return smoothed

def plot(data, label, color, title="Number of Crossing", weight=0.95, fpath='a', xlim=[]):
    sns.set()
    x_list = data[-1]
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlabel(f'Number of Node')

    i = 0
    while i < len(label):
        plt.plot(x_list, smooth(data[i], weight=weight), label=label[i], color = color[i], linewidth=1)
        i += 1

    locator = MultipleLocator(50)
    plt.gca().xaxis.set_major_locator(locator)
    
    plt.legend()
    plt.ylim(bottom=0)

    plt.savefig(f"{fpath}.png")

def test(csv_path = 'test.csv'):
    f = pd.read_csv(csv_path)
    plot(data=[f.avg1, f.avg2, f.name,], 
         label=['step=100', 'step=200', ], 
          color=['#00ff00','#ff00ff', '#ff0000', ], 
         title="time(s)", weight=0., fpath='time')
        

if __name__ == '__main__':
    test(csv_path = 'run_time.csv')