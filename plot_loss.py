import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
import pickle
import matplotlib.colors as mcolors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('-i', '--start-idx', type=int, default=2)
    args = parser.parse_args()

    models = [args.model] if args.model else os.listdir('models')

    c = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(5):
        c += c

    l = []
    for i, m in enumerate(models):
        hist = pickle.load(open(osp.join('models', m, 'history.pkl'), 'rb'))
        epochs = list(range(1 + args.start_idx, len(hist['loss']) + 1))
        p, = plt.plot(epochs, hist['loss'][args.start_idx:], color=c[i])
        l.append(p)
        plt.plot(epochs, hist['val_loss'][args.start_idx:], linestyle='--', color=c[i])
    plt.legend(l, models)
    plt.show()



