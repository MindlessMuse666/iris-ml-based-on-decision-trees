from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class KNNClassifierModel:
    '''
    Класс для обучения и оценки модели K-Nearest Neighbors.
    '''
    def __init__(self, n_neighbors=5):
        '''
        Конструктор класса.

        Args:
            n_neighbors (int): Количество соседей для классификации.
        '''
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


    def fit(self, X_train, y_train):
        '''
        Обучает модель KNN.

        Args:
            X_train: Обучающие данные.
            y_train: Метки классов для обучающих данных.
        '''
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        '''
        Прогнозирует метки классов для новых данных.

        Args:
            X_test: Тестовые данные.

        Returns:
            numpy.ndarray: Прогнозируемые метки классов.
        '''
        return self.model.predict(X_test)


    def evaluate(self, X_test, y_test):
        '''
        Оценивает производительность модели.

        Args:
            X_test: Тестовые данные.
            y_test: Истинные метки классов для тестовых данных.

        Returns:
            dict: Словарь с метриками (accuracy, precision, recall, f1).
        '''
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}