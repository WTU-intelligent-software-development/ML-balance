# -*-coding:utf-8-*-
# @Time   : 2022/10/16 10:30
# @Author : 王梓涵

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import det_curve, precision_recall_curve, average_precision_score
import numpy as np
def ROC(y_pred, y_test, figfilename, modelname):
    ################        ROC      ################
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_test, y_pred,pos_label=1)
    roc_auc[0] = auc(fpr[0], tpr[0])
    # Plot fig
    plt.plot(fpr[0], tpr[0],lw=2, label= modelname + ' (area = %0.2f)' % roc_auc[0])
    #set the style
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #set the x,y axis
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #set the font
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.title('RCO Fig', fontsize = fontsize)
    plt.legend()
    #save the fig as pdf file
    ##plt.savefig(figure_file + ".pdf")
    #save the fig as one png file
    plt.savefig(figfilename+"ROC"+".png")
    return roc_auc

def PR(y_pred, y_test, figfilename, modelname):
    ################        PR      ################
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, lw=2, label= modelname + ' (AP = %0.2f)' % average_precision_score(y_test, y_pred))
    #set the style
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.legend()
    plt.title('PR Fig', fontsize = fontsize)
    #save the fig as pdf file
    #plt.savefig(figure_file + ".pdf")
    #save the fig as one png file
    plt.savefig(figfilename+"PR"+".png")
    return average_precision_score(y_test, y_pred)

