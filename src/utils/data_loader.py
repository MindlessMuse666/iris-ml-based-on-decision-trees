from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size=0.2, random_state=42):
    '''
    Загружает датасет Iris и разделяет его на обучающую и тестовую выборки.

    Args:
        test_size (float): Размер тестовой выборки.
        random_state (int): Случайное состояние для воспроизводимости.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    '''
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_split_data()
    
    print('Размер обучающей выборки X:', X_train.shape)
    print('Размер тестовой выборки X:', X_test.shape)
    print('Названия признаков:', feature_names)
    print('Названия классов:', target_names)