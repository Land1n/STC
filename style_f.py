import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import math

class PlotManager:
    def __init__(self):
        self.set_default_styles()

    def set_default_styles(self):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12.0
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.labelsize'] = 'large'
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.9
        plt.rcParams['figure.subplot.bottom'] = 0.15

    def load_data(self, file_path=None, data=None, clipboard_data=None):
        """
        Универсальная загрузка данных:
        - из строки с табуляцией (clipboard_data),
        - из файла (file_path),
        - из DataFrame или словаря (data).
        Возвращает DataFrame или None, если данных нет.
        """
        if clipboard_data is not None and isinstance(clipboard_data, str) and clipboard_data.strip() != '':
            try:
                df = pd.read_csv(io.StringIO(clipboard_data), delimiter='\t', decimal=',', header=None)
                return df
            except Exception as e:
                raise ValueError(f"Ошибка при чтении данных из строки: {e}")

        if file_path is not None:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")

        if data is not None:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                try:
                    df = pd.DataFrame(data)
                    return df
                except Exception as e:
                    raise ValueError(f"Ошибка при преобразовании словаря в DataFrame: {e}")
            else:
                raise ValueError("Unsupported data format")

        return None  # Нет данных

    def plot(self, data=None, x_column=0, y_column=None, labels=None, title='', xlabel='', ylabel='',
             xlim=None, ylim=None, xscale='linear', yscale='linear',
             marker='circle', color='blue', linewidth=1.5,
             xstart=0, ystart=None, xticks=None, yticks=None,
             ask_input=False):
        """
        Построение графика.
        Если ask_input=True и data не передан, запрашивает ввод через консоль.
        Если x_column=None, создает автоматические точки по оси X (1, 2, 3, ...).
        labels - список или словарь с названиями для легенды.
        """

        # Запрос данных через консоль, если нужно
        if data is None and ask_input:
            print("Введите данные построчно (табуляция - разделитель столбцов).")
            print("Для окончания ввода нажмите Enter на пустой строке:")
            lines = []
            while True:
                line = input()
                if line.strip() == '':
                    break
                lines.append(line)
            clipboard_data = '\n'.join(lines)
            if clipboard_data.strip() == '':
                print("Нет данных для построения графика.")
                return
            try:
                data = pd.read_csv(io.StringIO(clipboard_data), delimiter='\t', decimal=',', header=None)
            except Exception as e:
                print(f"Ошибка при чтении данных: {e}")
                return

        # Если данных нет, завершаем
        if data is None:
            print("Данные не переданы и ask_input=False, построение невозможно.")
            return

        # Если данные - словарь, преобразуем в DataFrame
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                print(f"Ошибка при преобразовании словаря в DataFrame: {e}")
                return

        # Проверяем наличие столбца x_column или создаем автоматические точки
        if x_column is None:
            # Создаем автоматические точки по оси X (1, 2, 3, ...)
            max_len = max(len(data[col]) for col in data.columns)
            data['auto_x'] = np.arange(1, max_len + 1)
            x_column = 'auto_x'
        elif x_column not in data.columns:
            print(f"Ошибка: столбец '{x_column}' отсутствует в данных. Доступные столбцы: {list(data.columns)}")
            return

        # Преобразование данных в числовой формат
        data[x_column] = pd.to_numeric(data[x_column], errors='coerce')

        # Определяем столбцы Y
        if y_column is None:
            y_columns = [col for col in data.columns if col != x_column]
            if not y_columns:
                print("Нет столбцов для построения по оси Y.")
                return
        else:
            if y_column not in data.columns:
                print(f"Ошибка: столбец '{y_column}' отсутствует в данных.")
                return
            y_columns = [y_column]

        # Преобразуем столбцы Y в числовой формат
        for col in y_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Удаляем строки с NaN в x_column
        data = data.dropna(subset=[x_column])

        # Обработка labels для легенды
        if labels is None:
            labels_to_use = [str(col) for col in y_columns]
        elif isinstance(labels, dict):
            labels_to_use = [labels.get(col, str(col)) for col in y_columns]
        elif isinstance(labels, (list, tuple)):
            if len(labels) != len(y_columns):
                print("Ошибка: длина labels не совпадает с количеством графиков")
                return
            labels_to_use = labels
        else:
            print("Ошибка: labels должен быть списком, кортежем или словарём")
            return

        # Подготовка стилей
        color_styles = [
            'b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange', 'purple', 'saddlebrown', 'hotpink', 'gray'
        ]
        marker_styles = {
            'circle': 'o', 'square': 's', 'triangle_up': '^', 'triangle_down': 'v',
            'diamond': 'D', 'pentagon': 'p', 'star': '*', 'plus': 'P', 'x': 'X', 'point': '.'
        }
        marker_list = list(marker_styles.values())
        multiple = len(y_columns) > 1

        fig, ax = plt.subplots()

        for i, col in enumerate(y_columns):
            mask = data[col].notna()
            x_vals = data.loc[mask, x_column]
            y_vals = data.loc[mask, col]

            if multiple:
                clr = color_styles[i % len(color_styles)]
                mkr = marker_list[i % len(marker_list)]
            else:
                clr = color_styles[color_styles.index(color) % len(color_styles)] if color in color_styles else color
                mkr = marker_styles.get(marker, 'o')

            ax.plot(x_vals, y_vals,
                    marker=mkr,
                    color=clr,
                    linestyle='-',
                    linewidth=linewidth,
                    label=labels_to_use[i])

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # Автоматический подбор пределов осей
        if xlim is None:
            x_min, x_max = data[x_column].min(), data[x_column].max()
            x_range = x_max - x_min
            padding_x = x_range * 0.05 if x_range != 0 else 1
            ax.set_xlim(x_min - padding_x, x_max + padding_x)
        else:
            ax.set_xlim(xlim)

        if ylim is None:
            y_min = min(data[col].min() for col in y_columns)
            y_max = max(data[col].max() for col in y_columns)
            y_range = y_max - y_min
            padding_y = y_range * 0.05 if y_range != 0 else 1
            ax.set_ylim(y_min - padding_y, y_max + padding_y)
        else:
            ax.set_ylim(ylim)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        def nice_step(data_range, max_ticks=10):
            if data_range == 0:
                return 1
            raw_step = data_range / max_ticks
            magnitude = 10 ** math.floor(math.log10(raw_step))
            residual = raw_step / magnitude
            if residual < 1.5:
                step = 1 * magnitude
            elif residual < 3:
                step = 2 * magnitude
            elif residual < 7:
                step = 5 * magnitude
            else:
                step = 10 * magnitude
            return step

        # Настройка xticks
        if xticks is None:
            x_min, x_max = ax.get_xlim()
            step_x = nice_step(x_max - x_min)
            start_x = xstart if xstart is not None else math.ceil(x_min / step_x) * step_x
            ticks_x = np.arange(start_x, x_max + step_x, step_x)
            ax.set_xticks(ticks_x)
        else:
            x_min, x_max = ax.get_xlim()
            start_x = xstart if xstart is not None else x_min
            ax.set_xticks(np.arange(start_x, x_max + xticks, xticks))

        # Настройка yticks
        if yticks is None:
            y_min, y_max = ax.get_ylim()
            step_y = nice_step(y_max - y_min)
            start_y = ystart if ystart is not None else math.ceil(y_min / step_y) * step_y
            ticks_y = np.arange(start_y, y_max + step_y, step_y)
            ax.set_yticks(ticks_y)
        else:
            y_min, y_max = ax.get_ylim()
            start_y = ystart if ystart is not None else y_min
            ax.set_yticks(np.arange(start_y, y_max + yticks, yticks))

        ax.grid(True)
        plt.show()

"""
if __name__ == "__main__":
    pm = PlotManager()

    # Пример с передачей словаря
    data_dict = {
        #'x': [1,3,5,8,9],
        'y1': [2, 3, 5, 7, 11],
        'y2': [1, 4, 6, 8, 10]
    }
    df = pm.load_data(data=data_dict)

    # Построение графиков с автоматическими точками по оси X
    pm.plot(
        data=df, #None, для скопированных данных
        x_column=None,  # None, для автоматического расположения точек, 0 для скопированных данных
        y_column=None,
        labels={'y1': 'График 1', 'y2': 'График 2'},
        title='Графики с автоматической осью X',
        xlabel='Номер точки',
        ylabel='Значение',
        marker='triangle_up',
        color='blue',
        linewidth=1.5,
        xstart=0,
        ystart=None,
        xticks=None,
        yticks=None,
        ask_input=False #True, если надо вводить скопированные столбцы
    )
"""
    # Для запроса ввода через консоль:
    # pm.plot(x_column=None, ask_input=True)
