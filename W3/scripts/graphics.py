import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

dir_name = '../output/'
params_name = 'params.json'
results_name = 'results.json'

def plot_static_grayscale_alpha():
    """For the static modelling, plot on the X axis the alpha values and on the Y axis the mAP values.
    It reads this values from the output folder and plots the values."""
    dir_name = '../output/'
    dir_list = os.listdir('../output/')
    dir_list = [d for d in dir_list if '_static_grayscale_alpha' in d]
    alphas = []
    maps = []
    for d in dir_list:
        f = open(dir_name + d + '/' + params_name)
        params = json.load(f)
        alphas.append(params['ALPHA'])
        f.close()

        f = open(dir_name + d + '/' + results_name)
        results = json.load(f)
        maps.append(results['mAP of the video'])
        f.close()

    sns.lineplot(x=alphas, y=maps)
    plt.xlabel('Alpha')
    plt.ylabel('mAP')
    plt.show()

def plot_mAP_comparison(mAP_results, percentage):
    method_labels = [f"{method} (mAP: {mAP:.3f})" for method, _, mAP in mAP_results]
    mAP_values = [mAP for _, _, mAP in mAP_results]
    percentage_str = str(percentage*100)

    plt.bar(method_labels, mAP_values)
    plt.xlabel('Background Subtraction Method')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.title(f'Comparison of Mean Average Precision (mAP) for Different Background Subtraction Methods with Frame Percentage = {percentage_str}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 1)  
    plt.tight_layout()

    # Save the plot
    plt.savefig('./output/mAP_comparison_methods.png')
    plt.show()

def plot_adaptive_grayscale_alpha_rho():
    """For the adaptive modelling, plot on the X axis the alpha values and on the Y axis the mAP values.
    It reads this values from the output folder and plots the values."""
    dir_list = os.listdir('../output/')
    dir_list = [d for d in dir_list if '_adaptive_grayscale_alpha' in d]
    alphas = []
    rhos = []
    maps = []
    for d in dir_list:
        f = open(dir_name + d + '/' + params_name)
        params = json.load(f)
        alphas.append(params['ALPHA'])
        rhos.append(params['RHO'])
        f.close()

        f = open(dir_name + d + '/' + results_name)
        results = json.load(f)
        maps.append(results['mAP of the video'])
        f.close()

    # Plot the values
    sns.scatterplot(x=alphas, y=rhos, hue=maps, palette='rocket_r')
    # Put hue values in text above the points
    for i in range(len(alphas)):
        plt.text(alphas[i], rhos[i] + 0.009, round(maps[i], 2), ha='center')
    plt.xlabel('Alpha')
    plt.ylabel('Rho')
    plt.legend(title='mAP')
    plt.show()

plot_adaptive_grayscale_alpha_rho()
