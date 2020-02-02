import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GroupKFold
from collections import Counter, defaultdict


def Read_data():
    # Read iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Define ID
    list_id = ["A", "B", "C", "D", "E"]
    df["ID"] = np.random.choice(list_id, len(df))

    # Extract feature names
    features = iris.feature_names

    return df, features


def Count_y(y, groups):
    # y counts per group
    unique_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(unique_num))
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1

    return y_counts_per_group


def StratifiedGroupKFold(X, y, groups, features, k, seed = None):
    # Preparation
    max_y = np.max(y)
    y_counts_per_group = Count_y(y, groups)
    kf = GroupKFold(n_splits=k)

    for train_idx, val_idx in kf.split(X, y, groups):
        # Training dataset and validation dataset
        x_train = X.iloc[train_idx, :]
        id_train = x_train["ID"].unique()
        x_train = x_train[features]

        x_val, y_val = X.iloc[val_idx, :], y.iloc[val_idx]
        id_val = x_val["ID"].unique()
        x_val = x_val[features]

        # y counts of training dataset and validation dataset
        y_counts_train = np.zeros(max_y+1)
        y_counts_val = np.zeros(max_y+1)
        for id_ in id_train:
            y_counts_train += y_counts_per_group[id_]
        for id_ in id_val:
            y_counts_val += y_counts_per_group[id_]

        # Determination ratio of validation dataset
        numratio_train = y_counts_train / np.max(y_counts_train)
        stratified_count = np.ceil(y_counts_val[np.argmax(y_counts_train)] * numratio_train)
        stratified_count = stratified_count.astype(int)

        # Select validation dataset randomly
        val_idx = np.array([])
        np.random.seed(seed) 
        for num in range(max_y+1):
            val_idx = np.append(val_idx, np.random.choice(y_val[y_val==num].index, stratified_count[num]))
        val_idx = val_idx.astype(int)
        
        yield train_idx, val_idx


def Get_distribution(y_vals):
    # Get distribution
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())

    return [f"{y_distr[i] / y_vals_sum:.2%}" for i in range(np.max(y_vals) + 1)]

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

if __name__ == "__main__":
    
    df_iris, features = Read_data()

    print(df_iris.head())

    train_x = df_iris.drop("target", axis=1)
    train_y = df_iris["target"]
    groups = df_iris["ID"]

    distrs = [Get_distribution(train_y)]
    index = ["all dataset"]

    
    for fold, (train_idx, val_idx) in enumerate(StratifiedGroupKFold(X, y, groups, features, k=3)):

        print(f"TRAIN_ID - fold {fold}:", groups[train_idx].unique(), 
              f"TEST_ID - fold {fold}:", groups[val_idx].unique())
        
        distrs.append(Get_distribution(y[train_idx]))
        index.append(f"training set - fold {fold}")
        distrs.append(Get_distribution(y[val_idx]))
        index.append(f"validation set - fold {fold}")

    print(pd.DataFrame(distrs, index=index, columns=[f"Label {l}" for l in range(np.max(y) + 1)]))
