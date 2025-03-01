from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import graphviz


class DecisionTreeClassifierModel:
    '''
    Класс для обучения и оценки модели Decision Tree.
    '''
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=42):
        '''
        Конструктор класса.

        Args:
            criterion (str): Критерий для измерения качества разделения.
            max_depth (int): Максимальная глубина дерева.
            min_samples_split (int): Минимальное количество образцов, необходимых для разделения внутреннего узла.
            min_samples_leaf (int): Минимальное количество образцов, необходимых в листе узла.
            max_features (int): Максимальное количество атрибутов, рассматриваемых при делении узла.
            random_state (int):  Random state для воспроизводимости.
        '''
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        self.feature_names = None  #  Имена признаков для визуализации дерева
        self.target_names = None  # Имена классов для визуализации дерева


    def fit(self, X_train, y_train, feature_names=None, target_names=None):
        '''
        Обучает модель Decision Tree.

        Args:
            X_train: Обучающие данные.
            y_train: Метки классов для обучающих данных.
            feature_names: (опционально) Названия признаков.
            target_names: (опционально) Названия классов.
        '''
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names
        self.target_names = target_names


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


    def visualize_tree(self, filename=r'report\graphics\decision_tree.dot'):
        '''
        Визуализирует дерево решений и сохраняет его в файл .dot.

        Args:
            filename (str): Имя файла для сохранения визуализации.
        '''
        if self.feature_names is None:
            print('Необходимо передать feature_names при обучении для визуализации дерева.')
            return

        dot_data = tree.export_graphviz(
            self.model,
            out_file=None,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render(filename, view=False, format='png', cleanup=True)
        
        print(f'\nДерево решений визуализировано и сохранено в файл {filename}.png')