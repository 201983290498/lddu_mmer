from sklearn import metrics
import numpy as np
import torch
# import logging
global logger
def dense(y):
	label_y = []
	for i in range(len(y)):
		for j in range(len(y[i])):
			label_y.append(y[i][j])

	return label_y

def get_accuracy(y, y_pre):
	sambles = len(y)
	count = 0.0
	for i in range(sambles):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1

		count += float(y_true) / float(all_y)
	acc = float(count) / float(sambles)
	acc=round(acc,4)
	return acc


def get_metrics(y, y_pre):
	"""
	:param y:1071*6
	:param y_pre: 1071*6
	:return:
	"""
	y = y.cpu().detach().numpy()
	y_pre = y_pre.cpu().detach().numpy()
	acc = get_accuracy(y, y_pre)
	micro_f1 = metrics.f1_score(y, y_pre, average='micro')
	micro_precision = metrics.precision_score(y, y_pre, average='micro')
	micro_recall = metrics.recall_score(y, y_pre, average='micro')
	return micro_f1, micro_precision, micro_recall, acc


def getBinaryTensor(imgTensor, boundary = 0.21):
    one = torch.ones_like(imgTensor).fill_(1)
    zero = torch.zeros_like(imgTensor).fill_(0)
    return torch.where(imgTensor > boundary, one, zero)


def search_binary(pred_score, total_label, eval_threshold=None):
    if not (eval_threshold is None):
        total_pred = getBinaryTensor(pred_score, eval_threshold)
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
        logger.info("Eval Threshold, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f",
                    total_micro_f1, total_micro_precision, total_micro_recall, total_acc)
    best_f1,best_threshold = 0,0
    best_result = []
    for threshold in range(150, 300, 5):
        total_pred = getBinaryTensor(pred_score, threshold / 1000.0)
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
        if total_micro_f1 > best_f1:
            best_threshold = threshold
            best_f1 = total_micro_f1
            best_result = [total_micro_f1, total_micro_precision, total_micro_recall, total_acc]
    logger.info("Best Threshold, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f， threshold: %f",
                best_result[0], best_result[1], best_result[2], best_result[3], best_threshold / 100)
    return best_threshold / 1000.0