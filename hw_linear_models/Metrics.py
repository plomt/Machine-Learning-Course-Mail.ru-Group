import numpy as np

def road_to_confusion_matrix(y_true, y_predict, percent):
    class_1 = y_predict[:, -1]  # работаем на предсказание, что элемент примет значение 1, то есть берем второй столбец y_predict
    quantile = np.quantile(a=class_1, q=percent)  # получаем значение процентиля
    top_data = [(class_1[i], i) for i in range(len(class_1)) if class_1[i] >= quantile]  # формируем ТОПовую выборку
    indexes = [elem[1] for elem in top_data]  # запоминаем нахождение элементов

    top_data_binary = []
    for elem in top_data:  # бинаризируем нашу ТОП выборку
        if elem[0] >= percent:
            top_data_binary.append(1)
        else:
            top_data_binary.append(0)
    top_y_true = [y_true[i] for i in range(len(y_true)) if
                  i in indexes]  # формируем данные с которыми будем сравниваться
    # Confusion Matrix(dict)
    confusion_matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for i in range(len(top_data_binary)):
        if top_data_binary[i] == top_y_true[i] and top_y_true[i] == 1:
            confusion_matrix['TP'] += 1
        elif top_data_binary[i] == top_y_true[i] and top_y_true[i] == 0:
            confusion_matrix['TN'] += 1
        elif top_data_binary[i] != top_y_true[i] and top_y_true[i] == 0:
            confusion_matrix['FP'] += 1
        else:
            confusion_matrix['FN'] += 1
    return confusion_matrix

def accuracy_score(y_true, y_predict, percent=0.5):
    conf_m = road_to_confusion_matrix(y_true, y_predict, percent)
    accuracy = (conf_m['TP'] + conf_m['TN']) / (conf_m['TP'] + conf_m['TN'] + conf_m['FP'] + conf_m['FN'])
    return accuracy

def precision_score(y_true, y_predict, percent=0.5):
    conf_m = road_to_confusion_matrix(y_true, y_predict, percent)
    precision = conf_m['TP'] / (conf_m['TP'] + conf_m['FP'])
    return precision

def recall_score(y_true, y_predict, percent=0.5):
    conf_m = road_to_confusion_matrix(y_true, y_predict, percent)
    recall = conf_m['TP'] / (conf_m['TP'] + conf_m['FN'])
    return recall

def lift_score(y_true, y_predict, percent=0.5):
    conf_m = road_to_confusion_matrix(y_true, y_predict, percent)
    precision = conf_m['TP'] / (conf_m['TP'] + conf_m['FP'])
    lift = precision / ((conf_m['TP'] + conf_m['FN']) / (conf_m['TP'] + conf_m['FN'] + conf_m['FP'] + conf_m['TN']))
    return lift

def f1_score(y_true, y_predict, percent=0.5):
    conf_m = road_to_confusion_matrix(y_true, y_predict, percent)
    precision = conf_m['TP'] / (conf_m['TP'] + conf_m['FP'])
    recall = conf_m['TP'] / (conf_m['TP'] + conf_m['FN'])
    f1 = 2 * precision * recall / (precision + recall)
    return f1


file = np.loadtxt('D:/Datasets/HW2_labels.txt',  delimiter=',')
y_predict, y_true = file[:, :2], file[:, -1]
print(accuracy_score(y_true, y_predict, percent=0))
print(precision_score(y_true, y_predict, percent=0.5))
print(recall_score(y_true, y_predict, percent=0.5))
print(lift_score(y_true, y_predict, percent=0.5))
print(f1_score(y_true, y_predict, percent=0.5))
