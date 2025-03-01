import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestClassifierModel:
    '''
    Класс для обучения и оценки модели Random Forest.
    '''
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42):
        '''
        Конструктор класса.

        Args:
            n_estimators (int): Количество деревьев в лесу.
            criterion (str): Критерий для измерения качества разделения.
            max_depth (int): Максимальная глубина дерева.
            min_samples_split (int): Минимальное количество образцов, необходимых для разделения внутреннего узла.
            min_samples_leaf (int): Минимальное количество образцов, необходимых в листе узла.
            max_features (str): Максимальное количество атрибутов, рассматриваемых при делении узла.
            random_state (int):  Random state для воспроизводимости.
        '''
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        self.feature_names = None  # Имена признаков для визуализации важности признаков


    def fit(self, X_train, y_train, feature_names=None):
        '''
        Обучает модель Random Forest.

        Args:
            X_train: Обучающие данные.
            y_train: Метки классов для обучающих данных.
            feature_names: (опционально) Названия признаков.
        '''
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names


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


    def plot_feature_importance(self, filename=r'report\graphics\feature_importance.png'):
        '''
        Визуализирует важность признаков.

        Args:
            filename (str): Имя файла для сохранения визуализации.
        '''
        if self.feature_names is None:
            print('Необходимо передать feature_names при обучении для визуализации важности признаков.')
            return

        feature_importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6), num='Важность признаков в модели Random Forest')
        sns.barplot(x='importance', y='feature', hue='feature', data=feature_importance_df, palette='viridis', legend=False)
        plt.title('Важность признаков в модели Random Forest')
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        print(f'\nГрафик важности признаков сохранен в файл {filename}')