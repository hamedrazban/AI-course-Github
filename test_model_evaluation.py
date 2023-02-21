import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # test precision, recall and F1 score
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
#
# print('TP0=2, FP0=1, FN0=0')
# print('TP1=0, FP1=2, FN1=2')
# print('TP2=0, FP2=1, FN2=2')
#
# print('\nprecision (micro) theory: sum(TP_i)/(sum(TP_i)+sum(FP_i))=', 2/(2+4))
# print('recall (micro) theory: sum(TP_i)/(sum(TP_i)+sum(FN_i))=', 2/(2+4))
# print('precision (micro): ', precision_score(y_true, y_pred, average='micro'))
# print('recall (micro): ', recall_score(y_true, y_pred, average='micro'))
# print('F1 score (micro): ', f1_score(y_true, y_pred, average='micro'))
#
# print('\nprecision (macro) theory: (2/3+0+0)/3=', 2/9)
# print('recall (macro) theory: (1+0+0)/3=', 1/3)
# print('precision (macro): ', precision_score(y_true, y_pred, average='macro'))
# print('recall (macro): ', recall_score(y_true, y_pred, average='macro'))
# print('F1 score (macro): ', f1_score(y_true, y_pred, average='macro'))
#
# print('\nprecision (weighted) theory: (2/6)*(2/3)+0+0)=', 2/9)
# print('recall (weighted) theory: (2/6)*1+0+0)', 1/3)
# print('precision (weighted): ', precision_score(y_true, y_pred, average='weighted'))
# print('recall (weighted): ', recall_score(y_true, y_pred, average='weighted'))
# print('F1 score (weighted): ', f1_score(y_true, y_pred, average='weighted'))
#
# print(classification_report(y_true, y_pred))

#test confusion matrix
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=('0','1','2'))
disp.plot()
plt.show()
