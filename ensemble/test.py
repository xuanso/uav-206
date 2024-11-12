import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm  # 导入 tqdm

if __name__ == '__main__':

    score_dir = './test_scores'
    label = np.load('./label/pred.npy')
    label = label.argmax(axis=-1)
    
    for mod in os.listdir(score_dir):
        npy_path = os.path.join(score_dir, mod, 'epoch1_test_score.npy')
        pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                a = list(pickle.load(f).items())
                b = [i[1] for i in a]
                scores = np.array(b)
        scores = np.array(scores)
        all_zero = np.zeros(155)
        all_zero[97] = 1
        final_scores = np.insert(scores, 3222, all_zero, axis=0)
        pred = final_scores.argmax(axis=-1)
        acc = accuracy_score(label, pred)
        print(mod, acc)
       
    pred = np.load('pred.npy')
    pred = pred.argmax(axis=-1)
    acc = accuracy_score(label, pred)
    print(acc)