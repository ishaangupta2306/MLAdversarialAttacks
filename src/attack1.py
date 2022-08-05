# import argparse
# import os.path
# import warnings
#
# from typing import Optional, List, Tuple
# from tools.inception_v3_imagenet import model
#
# import numpy as np
# from sting.classifier import Classifier
# from sting.data import Feature, parse_c45, FeatureType
#
# # import util
#
#
# # In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# # variable names is lowercase_separated_by_underscores, unlike Java.
# class PartialInformationAttack(Classifier):
#     def __init__(self, schema: List[Feature], m=0):
#         self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
#         self._majority_label = 0  # Protected attributes in Python have an underscore prefix
#         self._class_occurences = np.array([0, 0])  # number of examples with each class label
#         self._feature_occurences = []  # feature_counts[feature][value, class] = #examples(x[feature] = value and y = class)
#         self.m = m
#         for f in schema:
#             if f.ftype == FeatureType.BINARY:
#                 self._feature_occurences.append(np.zeros((2, 2)))
#             elif f.ftype == FeatureType.NOMINAL:
#                 n = len(f.values)
#                 self._feature_occurences.append(np.zeros((n, 2)))
#             else:
#                 raise ValueError("Continuous features are not supported, please discretize before passing")
#
#     def attack(self, X: n, y: int):
#         epsilon = starting_perturbation
#         #xadv ← image of target class yadv
#         #xadv ← CLIP(xadv, x − , x + )
#
#     sess = tf.InteractiveSession()
#     eval_logits, eval_preds = model(sess, tf.expand_dims(x, 0))
#     def robust_in_top_k(t_, prop_adv_,k_):
#         if k == NUM_LABELS:
#             return True
#         for i in range(1):
#             n = np.random.rand(*prop_adv_.shape)*args.sigma
#             eval_logits_ = sess.run(eval_logits, {x: prop_adv_})[0]
#             if not t_ in eval_logits_.argsort()[-k_:][::-1]:
#                 return False
#         return True
#
#
#     # HERE IS THE IMAGE POINTER
#     image = open('your_image.png')
#
#     # CALLING THE FUNCTION
#     attack(image)
#     def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
#         """
#         This is the method where the training algorithm will run.
#         Args:
#             X: The dataset. The shape is (n_examples, n_features).
#             y: The labels. The shape is (n_examples,)
#             weights: Weights for each example. Will become relevant later in the course, ignore for now.
#         """
#         for c in [0, 1]:
#             # examples with class label = c
#             examples = X[y == c]
#             self._class_occurences[c] = examples.shape[0]
#
#             f_count = len(self._schema)
#             for feature in range(f_count):
#                 values, counts = np.unique(examples[:, feature], return_counts=True)
#                 for i in range(values.size):
#                     self._feature_occurences[feature][int(values[i])-1, c] = int(counts[i])
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         # basic naive version iterating through each example. Look into optimization later.
#         n_samples = X.shape[0]
#         f_count = len(self._schema)
#         y = np.zeros((n_samples, 2), dtype=np.float32)
#
#         for i in range(n_samples):
#             x = X[i]
#             x_likelihoods = [1, 1]
#             for f in range(f_count):
#                 n_values = len(self.schema[f].values)
#                 prior = 1/n_values
#                 m = n_values if self.m < 0 else self.m
#                 l0 = self._feature_occurences[f][int(x[f])-1, 0] + m * prior
#                 l0 = l0 / (self._class_occurences[0] + m)
#                 l1 = self._feature_occurences[f][int(x[f])-1, 1] + m * prior
#                 l1 = l1 / (self._class_occurences[1] + m)
#                 x_likelihoods[0] *= l0
#                 x_likelihoods[1] *= l1
#                 confidence = max(x_likelihoods[0], x_likelihoods[1])/((self._feature_occurences[f][int(x[f])-1, 1] + self._feature_occurences[f][int(x[f])-1, 0])/n_samples)
#             y[i][0] = 0 if x_likelihoods[0] > x_likelihoods[1] else 1  # argmax for binary class
#             y[i][1] = confidence
#         return y
#
#     @property
#     def schema(self):
#         """
#         Returns: The dataset schema
#         """
#         return self._schema
#
#     @property
#     def feature_occurences(self):
#         return self._feature_occurences
#
#
# accuracy = []
# precision = []
# recall = []
# auc_values = []
#
#
# def fetch_metric_data(classifier: NaiveBayes, X: np.ndarray, y: np.ndarray):
#     y_hat = classifier.predict(X)
#     accuracy.append(util.accuracy(y, y_hat[:, 0]))
#     precision.append(util.precision(y, y_hat[:, 0]))
#     recall.append(util.recall(y, y_hat[:, 0]))
#     sorted_confidences = util.sort_confidences(y, y_hat[:, 1])
#     auc_values.append(util.area_under_roc(sorted_confidences[:, 0], sorted_confidences[:, 1]))
#
#
# def evaluate_print_metrics():
#     # Calculating average values for metrics
#     accuracy_mean = np.mean(accuracy, dtype=np.float32)
#     precision_mean = np.mean(precision, dtype=np.float32)
#     recall_mean = np.mean(recall, dtype=np.float32)
#     auc_mean = np.mean(auc_values, dtype=np.float32)
#
#     # Calculating standard deviation for metrics
#     accuracy_std = np.std(accuracy, dtype=np.float32)
#     precision_std = np.std(precision, dtype=np.float32)
#     recall_std = np.std(recall, dtype=np.float32)
#
#     # Printing mean & standard deviation values of metrics
#     print("Accuracy: ", accuracy_mean, accuracy_std)
#     print("Precision: ", precision_mean, precision_std)
#     print("Recall: ", recall_mean, recall_std)
#     print("AUC ", auc_mean)
#
#
#     # Not implemented error specifically for area under ROC
#     #raise NotImplementedError
#
#
# def nbayes(data_path: str, bin_number: int, m: float, use_cross_validation: bool = True):
#     path = os.path.expanduser(data_path).split(os.sep)
#     file_base = path[-1]
#     root_dir = os.sep.join(path[:-1])
#     schema, X, y = parse_c45(file_base, root_dir)
#
#     # discretize continuous fetaures
#     # should we do this inside or outside the classifier?
#     for i in range(len(schema)):
#         feature = schema[i]
#         if schema[i].ftype == FeatureType.CONTINUOUS:
#             d_feature, d_values = util.split_bins(feature, X[:, i], bin_number)
#             schema[i] = d_feature
#             X[:, i] = d_values
#
#     if use_cross_validation:
#         datasets = util.cv_split(X, y, folds=5, stratified=True)
#     else:
#         datasets = ((X, y, X, y),)
#
#     for X_train, y_train, X_test, y_test in datasets:
#         classifier = NaiveBayes(schema, m=m)
#         classifier.fit(X_train, y_train)
#         fetch_metric_data(classifier, X_test, y_test)
#     evaluate_print_metrics()
#
#
# def calculate_nb_parameters(schema, X, y):
#     np.concatenate((X, y.T), axis=1)
#     for i in range(len(schema)):
#         for nominal_values in schema[i].NominalValues:
#             true_count_by_feature_value = {[schema[i], nominal_values]: ((X[:, i] == nominal_values) & (X[:, -1] == 1)).sum()}
#             false_count_by_feature_value = {[schema[i], nominal_values]: ((X[:, i] == nominal_values) & (X[:, -1] == 0)).sum()}
#
#
# if __name__ == '__main__':
#     # Set up argparse arguments
#     parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
#     parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
#     parser.add_argument('bin_number', metavar='bin_number', type=int,
#                         help='Bin number its sorted into, must be greater than two')
#     parser.add_argument('m', metavar='M', type=float,
#                         help='M for for the m-estimate')
#     parser.add_argument('--no-cv', dest='cv', action='store_false',
#                         help='Disables cross validation and trains on the full dataset.')
#
#     parser.set_defaults(cv=True)
#     args = parser.parse_args()
#
#     # If the depth limit is negative throw an exception
#     if args.bin_number < 2:
#         raise argparse.ArgumentTypeError('Bin number must be greater than two')
#
#     # You can access args with the dot operator like so:
#     data_path = os.path.expanduser(args.path)
#     use_cross_validation = args.cv
#     bin_number = args.bin_number
#     m_value = args.m
#     nbayes(data_path, bin_number, m_value, use_cross_validation)
