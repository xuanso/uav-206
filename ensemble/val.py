import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm  # 导入 tqdm

if __name__ == '__main__':

    score_dir = './val_scores'
    label = np.load('./label/val_label.npy')
    scores = []
    for mod in os.listdir(score_dir):
        pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                a = list(pickle.load(f).items())
                b = [i[1] for i in a]
                score = np.array(b)
        score = np.array(score)
        scores.append(score)

        pred = score.argmax(axis=-1)
        acc = accuracy_score(label, pred)
        print(mod, acc)
        
    scores = np.array(scores)
    
    #alpha = [0.473, 0.362, 0.514, 0.542, 0.511, 0.637, 0.41,  0.555, 0.419, 0.402, 0.411]
    alpha = [0.24, 0.14, 0.67, 0.47, 0.82, 0.98, 0.13,  0.84, 0.44, 0.24, 0.59]
    #alpha = [1]*len(scores)
    pred_scores = np.zeros([2000, 155])
    for i in range(len(scores)):
        pred_scores += alpha[i] * scores[i]
    pred = pred_scores.argmax(axis=-1)
    acc = accuracy_score(label, pred )

    print(acc)