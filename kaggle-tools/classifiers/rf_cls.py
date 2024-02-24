# Random Forest From Scratch
import numpy as np
import random
import pandas as pd 
import sys

sys.path.append('/Users/albertluo/version-control/kaggle/kaggle-tools/')

from kaggle_eda_tool import drop_correlation, sl

class RandomForestClassifier:
  def __init__(self, n_estimators, max_features, max_depth, min_samples_split):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

  def entropy(self, p):
    if p == 0 or p == 1:
      return 0
    
    else:
      return -(p * np.log2(p) * (1 - p) + np.log2(1 - p))

  def information_gain(self, left_child, right_child, target_value):
    parent = left_child + right_child
    p_parent = parent.count(target_value) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(target_value) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(target_value) / len(right_child) if len(right_child) > 0 else 0
    IG_p = self.entropy(p_parent)
    IG_l = self.entropy(p_left)
    IG_r = self.entropy(p_right)

    return IG_p - (len(left_child) / len(parent)) * IG_l - (len(right_child) / len(parent)) * IG_r

  def bootstrap(self, X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]

    return X_bootstrap, y_bootstrap, X_oob, y_oob

  def oob_score(self, tree, X_test, y_test):
    mislabel = 0

    for i in range(len(X_test)):
      pred = self.predict_tree(tree, X_test[i])
      if pred != y_test[i]:
        mislabel += 1

      return mislabel / len(X_test) 

  def find_split_point(self, X_bootstrap, y_bootstrap):
    feature_list = list()
    num_features = len(X_bootstrap[0])

    while len(feature_list) <= self.max_features:
      feature_index = random.sample(range(num_features), 1)
      if feature_index not in feature_list:
        feature_list.extend(feature_index)
      
      best_info_gain = -999
      node = None

      for feature_index in feature_list:
        for split_point in X_bootstrap[:, feature_index]:
          left_child = {'X_bootstrap' : [], 'y_bootstrap' : []}
          right_child = {'X_bootstrap' : [], 'y_bootstrap' : []}

          if type(split_point) in [float, int]:
            for i, value in enumerate(X_bootstrap[:, feature_index]):
              if value <= split_point:
                left_child['X_bootstrap'].append(X_bootstrap[i])
                left_child['y_bootstrap'].append(y_bootstrap[i])
              else:
                right_child['X_bootstrap'].append(X_bootstrap[i])
                left_child['y_bootstrap'].append(y_bootstrap[i])
              
          else:
            for i, value in enumerate(X_bootstrap[:, feature_index]):
              if split_point == value:
                left_child['X_bootstrap'].append(X_bootstrap[i])
                left_child['y_bootstrap'].append(y_bootstrap[i])
              else:
                right_child['X_bootstrap'].append(X_bootstrap[i])
                right_child['y_bootstrap'].append(y_bootstrap[i])
          
          split_info_gain = self.information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'], target_value=1)

          if split_info_gain > best_info_gain:
            best_info_gain = split_info_gain
            left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
            right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
            node = {'information_gain' : split_info_gain, 
                    'left_child' : left_child, 
                    'right_child' : right_child, 
                    'split_point' : split_point, 
                    'feature_index' : feature_index}

    return node

  def terminal_node(self, node):
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred

  def split_node(self, node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if (len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0):
      empty_child = {'y_bootstrap' : left_child['y_bootstrap'] + right_child['y_bootstrap']}

      node['left_split'] = self.terminal_node(empty_child)
      node['right_split'] = self.terminal_node(empty_child)

      return 

    if depth >= max_depth:
      node['left_split'] = self.terminal_node(left_child)
      node['right_split'] = self.terminal_node(right_child)
      
      return

    if (len(left_child['y_bootstrap']) <= min_samples_split).any():
      node['left_split'] = node['right_split'] = self.terminal_node(left_child)

    else:
      node['left_split'] = self.find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'])
      self.split_node(node['left_split'], max_features, min_samples_split, max_depth, depth + 1)

    if ((len(right_child['y_bootstrap'])) <= min_samples_split).any():
      node['left_split'] = node['right_split'] = self.terminal_node(right_child)
    else:
      node['right_split'] = self.find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'])
      self.split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

  def build_tree(self, X_bootstrap, y_bootstrap):
    root_node = self.find_split_point(X_bootstrap, y_bootstrap)
    self.split_node(root_node, X_bootstrap, y_bootstrap, self.max_depth, depth=1)
    return root_node

  def random_forest_classifier(self, X_train, y_train):
    self.trees = []
    oobs = []

    for _ in range(self.n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = self.bootstrap(X_train, y_train)
        tree = self.build_tree(X_bootstrap, y_bootstrap)
        self.trees.append(tree)

        oob_error = self.oob_score(tree, X_oob, y_oob)
        oobs.append(oob_error)

    print("OOB Estimate: {:.2f}".format(np.mean(oobs)))

  def predict_tree(self, tree, X_test):
    feature_index = tree['feature_index']

    if X_test[feature_index] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return self.predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return self.predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']
  
  def predict_rf(self, X_test):
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [self.predict_tree(tree, X_test.values[i]) for tree in self.trees]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

# preprocessing:
DF = pd.read_csv('/Users/albertluo/version-control/kaggle/kaggle-tools/breast_cancer/breast-cancer.csv')
DF['diagnosis'] = DF['diagnosis'].map({'M': 0, 'B': 1})
# drop_correlation(DF, 0.90, 4)

# testing on data:
# nb_train = int(np.floor(0.8 * len(DF)))
# features = list(DF.columns[2:])
# DF = DF.sample(frac = 1, random_state = 42)
# X_train = DF[features][:nb_train]
# y_train = DF['diagnosis'][:nb_train].values
# X_test = DF[features][nb_train:]
# y_test = DF['diagnosis'][nb_train:].values

# N_ESTIMATORS = 3
# MAX_FEATURES = 20
# MAX_DEPTH = 5
# MIN_SAMPLES_SPLIT = 2

# rf_classifier = RandomForestClassifier(
#     n_estimators=N_ESTIMATORS,
#     max_features=MAX_FEATURES,
#     max_depth=MAX_DEPTH,
#     min_samples_split=MIN_SAMPLES_SPLIT
# )

# rf_classifier.random_forest_classifier(X_train, y_train)

# # Predict and evaluate the classifier
# preds = rf_classifier.predict_rf(X_test)
# acc = sum(preds == y_test) / len(y_test)
# print('Testing Accuracy: {}'.format(np.round(acc, 4)))