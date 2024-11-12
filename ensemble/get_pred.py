import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm  # 导入 tqdm

if __name__ == '__main__':
    # test_B
    
    score_dir = './test_scores'
    scores = []
    mod_names = []
    for mod in os.listdir(score_dir):
        mod_names.append(mod)
        npy_path = os.path.join(score_dir, mod, 'epoch1_test_score.npy')
        pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                a = list(pickle.load(f).items())
                b = [i[1] for i in a]
                scores.append(np.array(b))
        elif os.path.exists(npy_path):  # 如果没有 .pkl，则检查 .npy
            scores.append(np.load(npy_path))
        else:
            print(f"Neither {pkl_path} nor {npy_path} exists for model {mod}.")
    scores = np.array(scores)
    
    print(mod_names)
    #[0.473, 0.362, 0.514, 0.542, 0.511, 0.637, 0.41,  0.555, 0.419, 0.402, 0.411]
    alpha = [0.24, 0.14, 0.67, 0.47, 0.82, 0.98, 0.13,  0.84, 0.44, 0.24, 0.59]
    pred_scores = np.zeros([4306, 155])
    for i, _ in enumerate(alpha):
        pred_scores += alpha[i] * scores[i]

    all_zero = np.zeros(155)
    all_zero[98] = 1
    final_scores = np.insert(pred_scores, 802, all_zero, axis=0)
    print(final_scores.shape)
    np.save('pred.npy', final_scores)