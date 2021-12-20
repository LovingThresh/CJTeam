import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(save_path, predict_array, size=(512, 512)):
    if predict_array.ndim == 4:
        h_map = sns.heatmap(predict_array[:, :, :, 0].reshape(size), xticklabels=False, yticklabels=False)
    else:
        h_map = sns.heatmap(predict_array.reshape(size), xticklabels=False, yticklabels=False)
    plt.savefig(save_path, h_map)
