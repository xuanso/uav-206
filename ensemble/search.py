import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm  # 导入 tqdm


if __name__ == '__main__':

    avg_acc = 0
    avg_acc_suf = 0
    score_dir = './val_scores'
    mod_names = []
    for mod in os.listdir(score_dir):
            mod_names.append(mod)
    print(mod_names)

    score_dir = './val_scores'
    scores = []
    for mod in os.listdir(score_dir):
        mod_names.append(mod)
        pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                a = list(pickle.load(f).items())
                b = [i[1] for i in a]
                scores.append(np.array(b))
    scores = np.array(scores)


    label = np.load('./label/val_label.npy')

    pred_scores = np.zeros([2000, 155])
    for i in range(len(scores)):
        pred_scores +=  scores[i]
    pred = pred_scores.argmax(axis=-1)
    acc = accuracy_score(label, pred)

    print(acc)
    np.random.seed(None)
    seeds = np.random.randint(0, 2 ** 10, size=100)

    final_alpha=[0]*len(scores)
    
    for seed in seeds:
        np.random.seed(seed)

        #随机搜索

        best_acc = 0
        best_params = None
        num_iterations = 5  # 随机搜索次数


        for _ in tqdm(range(num_iterations), desc="Random Searching"):
            # 生成 15 个在 [0, 2) 范围内的随机数，并将其转化为 0.1 的倍数
            alpha = np.random.randint(0, 10, size=scores.shape[0]) * 0.1  # 生成 0.1 的倍数
            # alpha = [1]*scores.shape[0]
            # 计算加权得分
            pred_scores = np.zeros([2000, 155])
            for i in range(len(alpha)):
                pred_scores += alpha[i] * scores[i]

            pred = pred_scores.argmax(axis=-1)
            acc = accuracy_score(label, pred)
            # 更新最佳参数和准确率
            if acc > best_acc:
                best_acc = acc
                best_params = alpha

        print('Best accuracy:', best_acc)
        print('Best parameters:', best_params)

        # pred_scores = np.zeros([2000, 155])
        # for i in range(len(alpha)):
        #     pred_scores += alpha[i] * scores[i]
        #
        # pred = pred_scores.argmax(axis=-1)
        # acc_suf = accuracy_score(label[index2], pred[index2])
        # avg_acc_suf += acc_suf

        avg_acc+=best_acc
        final_alpha+=best_params


    avg_acc /= len(seeds)
    print('avg_acc:', avg_acc)

    # avg_acc_suf /=len(seeds)
    # print('avg_acc_suf:', avg_acc_suf)

    final_alpha /= len(seeds)
    print('final_alpha:', final_alpha)

    pred_scores = np.zeros([2000, 155])
    for i in range(len(alpha)):
        pred_scores += final_alpha[i] * scores[i]
    pred = pred_scores.argmax(axis=-1)
    final_acc = accuracy_score(label, pred)

    print('final_acc', final_acc)

