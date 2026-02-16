import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def paint_distribution(name, data, sample_sizes):
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6))
    
    boxplot_color = '#3498db'
    for i, size in enumerate(sample_sizes):
        ax = axes[i]
        ax.boxplot(data[i], patch_artist=True, boxprops=dict(facecolor=boxplot_color))
        
        ax.set_title(f'Размер выборки: {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Выборка', fontsize=11)
        ax.set_ylabel('Значение', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    fig.suptitle(f'{name} распределение', fontsize=18, fontweight='bold', y=1.0)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()
    
def count_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return len(outliers)

def main():
    random = np.random.default_rng(5)

    data = { "Нормальное": [], "Коши": [], "Лапласа": [], "Пуассона": [], "Равномерное": [] }
    sample_sizes = [20, 100]
    n_simulations = 1000
    outliers_count_for_simulations = {name: {size: [] for size in sample_sizes} for name in data.keys()}
    for j in range(n_simulations):
        for i in sample_sizes:
            data["Нормальное"].append(random.normal(0, 1, i))
            data["Коши"].append(random.standard_cauchy(i))
            data["Лапласа"].append(random.laplace(0, 1/np.sqrt(2), i))
            data["Пуассона"].append(random.poisson(10, i))
            data["Равномерное"].append(random.uniform(-np.sqrt(3), np.sqrt(3), i))
            for name in data:
                outliers_count_for_simulations[name][i].append(count_outliers(data[name][-1]))
        if(j==1):
            for name in data:
                paint_distribution(name, data[name], sample_sizes)
                
    for name in data:
        print(f"{name} распределение:")
        for size in sample_sizes:
            mean_outliers = np.mean(np.array(outliers_count_for_simulations[name][size])/size)
            print(f"Средняя доля выбросов для размера {size}: {mean_outliers:.2f}")
        

if __name__ == "__main__":
    main()