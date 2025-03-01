from utils.data_loader import load_and_split_data
from models.decision_tree import DecisionTreeClassifierModel
from models.random_forest import RandomForestClassifierModel
from models.knn import KNNClassifierModel
from visualization.visualizer import Visualizer
from sklearn.model_selection import cross_validate


def main():
    '''
    Основная функция для запуска обучения, оценки и визуализации моделей с кросс-валидацией.
    '''
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_split_data(test_size=0.3, random_state=42) # Увеличиваем тестовую выборку

    # 1. Обучение и оценка Decision Tree
    print('Обучение и оценка Decision Tree...')
    dt_model = DecisionTreeClassifierModel(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train, feature_names=feature_names, target_names=target_names)
    dt_metrics = dt_model.evaluate(X_test, y_test) #Оцениваем на отложенной тестовой выборке
    print('Метрики Decision Tree на тестовой выборке:', dt_metrics)

    # Кросс-валидация Decision Tree
    dt_cv_results = cross_validate(dt_model.model, X_train, y_train, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    print('Результаты кросс-валидации Decision Tree:', {
        'accuracy': dt_cv_results['test_accuracy'].mean(),
        'precision': dt_cv_results['test_precision_weighted'].mean(),
        'recall': dt_cv_results['test_recall_weighted'].mean(),
        'f1': dt_cv_results['test_f1_weighted'].mean()
    })
    dt_model.visualize_tree()

    # 2. Обучение и оценка Random Forest
    print('Обучение и оценка Random Forest...')
    rf_model = RandomForestClassifierModel(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train, feature_names=feature_names)
    rf_metrics = rf_model.evaluate(X_test, y_test) #Оцениваем на отложенной тестовой выборке
    print('Метрики Random Forest на тестовой выборке:', rf_metrics)

    # Кросс-валидация Random Forest
    rf_cv_results = cross_validate(rf_model.model, X_train, y_train, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    print('Результаты кросс-валидации Random Forest:', {
        'accuracy': rf_cv_results['test_accuracy'].mean(),
        'precision': rf_cv_results['test_precision_weighted'].mean(),
        'recall': rf_cv_results['test_recall_weighted'].mean(),
        'f1': rf_cv_results['test_f1_weighted'].mean()
    })
    rf_model.plot_feature_importance()

    # 3. Обучение и оценка KNN
    print('Обучение и оценка KNN...')
    knn_model = KNNClassifierModel(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_metrics = knn_model.evaluate(X_test, y_test) #Оцениваем на отложенной тестовой выборке
    print('Метрики KNN на тестовой выборке:', knn_metrics)

    # Кросс-валидация KNN
    knn_cv_results = cross_validate(knn_model.model, X_train, y_train, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    print('Результаты кросс-валидации KNN:', {
        'accuracy': knn_cv_results['test_accuracy'].mean(),
        'precision': knn_cv_results['test_precision_weighted'].mean(),
        'recall': knn_cv_results['test_recall_weighted'].mean(),
        'f1': knn_cv_results['test_f1_weighted'].mean()
    })

    # 4. Сравнение моделей (только результаты кросс-валидации)
    print('Сравнение моделей (результаты кросс-валидации)...')
    model_metrics = {
        'Decision Tree (CV)': {
            'accuracy': dt_cv_results['test_accuracy'].mean(),
            'precision': dt_cv_results['test_precision_weighted'].mean(),
            'recall': dt_cv_results['test_recall_weighted'].mean(),
            'f1': dt_cv_results['test_f1_weighted'].mean()
        },
        'Random Forest (CV)': {
            'accuracy': rf_cv_results['test_accuracy'].mean(),
            'precision': rf_cv_results['test_precision_weighted'].mean(),
            'recall': rf_cv_results['test_recall_weighted'].mean(),
            'f1': rf_cv_results['test_f1_weighted'].mean()
        },
        'KNN (CV)': {
            'accuracy': knn_cv_results['test_accuracy'].mean(),
            'precision': knn_cv_results['test_precision_weighted'].mean(),
            'recall': knn_cv_results['test_recall_weighted'].mean(),
            'f1': knn_cv_results['test_f1_weighted'].mean()
        }
    }
    visualizer = Visualizer()
    visualizer.plot_model_comparison(model_metrics, filename='model_comparison_cv.png')


if __name__ == '__main__':
    main()