import matplotlib.pyplot as plt


def plot_acc_per_class(acc_per_class: dict, save_path=None, relative=False):
    plt.figure(figsize=(12, 6))
    if not relative:
        plt.bar(acc_per_class.keys(), acc_per_class.values())
    else:
        for i, v in zip(acc_per_class.keys(), acc_per_class.values()):
            if v > 0:
                plt.bar(i, v, color='g')
            else:
                plt.bar(i, v, color='r')

    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    if not relative:
        plt.title('Accuracy per class')
    else:
        plt.title('Relative improvement per class')
    if not relative:
        plt.ylim(0, 1)
    else:
        plt.ylim(-1, 1)
    plt.subplots_adjust(bottom=0.3)
    if save_path is not None:
        plt.savefig(save_path)
    return plt
