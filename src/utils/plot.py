import os
from re import I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_train_summary(data, out_path):
    os.makedirs(out_path, exist_ok=True)

    plt.plot(data['G_losses_epoch'], label='G loss')
    plt.plot(data['D_losses_epoch'], label='D loss')

    if 'term_1_epoch' in data:
        plt.plot(np.array(data['term_1_epoch']), label="term_1")

    if 'term_2_epoch' in data:
        plt.plot(np.array(data['term_2_epoch']), label="term_2")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss.png'))
    plt.clf()

    plt.plot(data['D_x_epoch'], label='D(x)')
    plt.plot(data['D_G_z1_epoch'], label='D(G(z)) 1')
    plt.plot(data['D_G_z2_epoch'], label='D(G(z)) 2')
    plt.xlabel('epoch')
    plt.ylabel('output')
    plt.title('d outputs')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'd_outputs.png'))
    plt.clf()

    plt.plot(data['D_acc_real_epoch'], label='D acc real')
    plt.plot(data['D_acc_fake_1_epoch'], label='D acc fake 1')
    plt.plot(data['D_acc_fake_2_epoch'], label='D acc fake 2')
    plt.xlabel('epoch')
    plt.ylabel('output')
    plt.title('d accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'd_accuracy.png'))
    plt.clf()

    plt.plot(data['fid'], label='FID')
    plt.xlabel('epoch')
    plt.ylabel('fid')
    plt.title('fid')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'fid.png'))
    plt.clf()

    if 'focd' in data:
        plt.plot(data['focd'], label='F*D')
        plt.xlabel('epoch')
        plt.ylabel('f*d')
        plt.title('f*d')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'f*d.png'))
        plt.clf()

    if 'conf_dist' in data:
        plt.plot(data['conf_dist'], label='conf_dist')
        plt.xlabel('epoch')
        plt.ylabel('conf_dist')
        plt.title('conf_dist')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'conf_dist.png'))
        plt.clf()


def plot_metrics(data, path, C_name):
    fid = data["fid"].to_numpy()
    cd = data["conf_dist"].to_numpy()

    costs = np.array([c for c in zip(fid, cd)])
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1)
            is_efficient[i] = True

    size = [1.5 if pe else 1 for pe in is_efficient]
    data["pareto_efficient"] = is_efficient

    data.to_csv(os.path.join(path, f'metrics_{C_name}.csv'))

    sns.scatterplot(data=data, x="fid",
                    y="conf_dist", hue="weight", style="s1_epochs", palette="deep", size=size)
    plt.savefig(os.path.join(path, f'metrics_{C_name}.svg'))

    plt.close()
