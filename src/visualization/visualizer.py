import matplotlib.pyplot as plt


class Visualizer:
    '''
    Класс для создания визуализаций.
    '''
    def plot_model_comparison(self, model_metrics, filename='report\graphics\model_comparison.png'):
        '''
        Создает столбчатую диаграмму для сравнения метрик разных моделей.

        Args:
            model_metrics (dict): Словарь, где ключи - названия моделей, а значения - словари с метриками.
            filename (str): Имя файла для сохранения графика.
        '''
        model_names = list(model_metrics.keys())
        accuracy_scores = [model_metrics[model]['accuracy'] for model in model_names]
        precision_scores = [model_metrics[model]['precision'] for model in model_names]
        recall_scores = [model_metrics[model]['recall'] for model in model_names]
        f1_scores = [model_metrics[model]['f1'] for model in model_names]

        x = range(len(model_names))
        width = 0.2

        plt.figure(figsize=(12, 8), num='График сравнения моделей')
        plt.bar(x, accuracy_scores, width=width, label='Accuracy')
        plt.bar([i + width for i in x], precision_scores, width=width, label='Precision')
        plt.bar([i + 2 * width for i in x], recall_scores, width=width, label='Recall')
        plt.bar([i + 3 * width for i in x], f1_scores, width=width, label='F1 Score')

        plt.xticks([i + 1.5 * width for i in x], model_names)
        plt.ylabel('Значение')
        plt.title('Сравнение моделей')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        print(f'График сравнения моделей сохранен в файл {filename}')