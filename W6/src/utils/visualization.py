import matplotlib.pyplot as plt


def plot_acc_per_class(acc_per_class: dict, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.bar(acc_per_class.keys(), acc_per_class.values())
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Accuracy per class')
    plt.ylim(0, 1)
    plt.subplots_adjust(bottom=0.3)
    if save_path is not None:
        plt.savefig(save_path)
    return plt
