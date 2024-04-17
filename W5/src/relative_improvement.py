from utils.visualization import plot_acc_per_class
import json

if __name__ == '__main__':
    path_1 = 'results/eager-deluge-37_hmdb51/acc_per_class.json'
    path_2 = 'results/upbeat-plasma-11_hmdb51/acc_per_class.json'

    with open(path_1, 'r') as f:
        acc_per_class_1 = json.load(f)
    with open(path_2, 'r') as f:
        acc_per_class_2 = json.load(f)

    relative_improvement = {}
    for k in acc_per_class_1.keys():
        relative_improvement[k] = acc_per_class_2[k] - acc_per_class_1[k]

    # Order the relative improvement dictionary by value
    relative_improvement = dict(sorted(relative_improvement.items(), key=lambda item: item[1], reverse=True))

    plt = plot_acc_per_class(relative_improvement, relative=True)
    plt.show()