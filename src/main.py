from data_collector import get_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from naive_bayes import run_class

white, red = get_data()
dataset = white
w_tp_nb, w_fp_nb, w_tn_nb, w_fn_nb, w_accuracy_nb_train, w_accuracy_nb_test, w_cv_nb= run_class(dataset)
dataset = red
r_tp_nb, r_fp_nb, r_tn_nb, r_fn_nb, r_accuracy_nb_train, r_accuracy_nb_test, r_cv_nb= run_class(dataset)
models = [('White wine Bayes', w_tp_nb, w_fp_nb, w_tn_nb, w_fn_nb, w_accuracy_nb_train, w_accuracy_nb_test, w_cv_nb.mean()),
          ('Red wine Bayes', r_tp_nb, r_fp_nb, r_tn_nb, r_fn_nb, r_accuracy_nb_train, r_accuracy_nb_test, r_cv_nb.mean())]
predict = pd.DataFrame(data=models, columns=['Model', 'True Positive', 'False Positive', 'True Negative',
                                             'False Negative', 'Accuracy(training)', 'Accuracy(test)',
                                             'Cross-Validation'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
print(predict)

f, axe = plt.subplots(1, 1, figsize=(18, 6))

predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

sns.barplot(x='Cross-Validation', y='Model', data=predict, ax=axe)
# axes[0].set(xlabel='Region', ylabel='Charges')
axe.set_xlabel('Cross-Validaton Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0, 1.0)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()
