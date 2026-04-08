import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse

def print_results_table(final_data, dist_name):
    print(f"\n" + "="*60)
    print(f"РАСПРЕДЕЛЕНИЕ: {dist_name.upper()}")
    print("="*60)
    
    for ct in final_data.keys():
        print(f"\nКоэффициент: {ct}")
        
        sample_sizes = sorted(final_data[ct]["Среднее"][dist_name].keys())
        first_size = sample_sizes[0]
        recorded_rhos = sorted(final_data[ct]["Среднее"][dist_name][first_size].keys())
        
        header_rho = "n".ljust(5)
        header_md = "".ljust(5)
        
        for rho in recorded_rhos:
            header_rho += f"rho = {rho}".center(25)
            header_md += "M".center(12) + "D".center(13)
        
        print(header_rho)
        print(header_md)
        print("-" * len(header_rho))

        for size in sample_sizes:
            row_str = f"{size:<5}"
            for rho in recorded_rhos:
                m = final_data[ct]["Среднее"][dist_name][size][rho]
                d = final_data[ct]["Дисперсия"][dist_name][size][rho]
                row_str += f"{m:12.3f}{d:13.3f}"
            print(row_str)

def calc_quadratic_correlation(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    x = x - med_x
    y = y - med_y
    n1 = np.sum((x > 0) & (y > 0))
    n2 = np.sum((x < 0) & (y > 0))
    n3 = np.sum((x < 0) & (y < 0))
    n4 = np.sum((x > 0) & (y < 0))

    rq = (n1 + n3 - n2 - n4) / len(x)
    return rq

def plot_subplots_ellipses(size, min_vals, max_vals, rho_keys, title_prefix, n_std=3.0):
    fig, axes = plt.subplots(1, len(rho_keys), figsize=(5 * len(rho_keys), 5))
    
    if len(rho_keys) == 1:
        axes = [axes]
        
    colors = {0: 'blue', 0.5: 'green', 0.9: 'red', 'mix': 'purple'}
    
    for ax, rho in zip(axes, rho_keys):
        x = (min_vals[size][rho]['x'] + max_vals[size][rho]['x']) / 2
        y = (min_vals[size][rho]['y'] + max_vals[size][rho]['y']) / 2
        
        current_color = colors.get(rho, 'blue')
        
        ax.scatter(x, y, s=15, alpha=0.3, color=current_color)
        
        cov = np.cov(x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        theta = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        center_x, center_y = np.mean(x), np.mean(y)
        
        ellipse = Ellipse(
            xy=(center_x, center_y),
            width=width,
            height=height,
            angle=theta,
            facecolor='none',   
            edgecolor=current_color,    
            linestyle='-',     
            linewidth=2.5,
            label=f'Эллипс ({n_std} std)'
        )
        ax.add_patch(ellipse)
        ax.plot(center_x, center_y, marker='x', color=current_color, markersize=8)

        ax.axhline(0, color='black', lw=0.8, alpha=0.3)
        ax.axvline(0, color='black', lw=0.8, alpha=0.3)
        
        ax.set_aspect('equal', 'datalim')
        
        dist_name = "Смешанное" if rho == "mix" else f"ρ = {rho}"
        ax.set_title(dist_name)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')
        
    fig.suptitle(f"{title_prefix} (Размер выборки n={size})", fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.show()
    
def main():
    random = np.random.default_rng(4)
    
    dist_names = ["Нормальное", "Смешанное"]
    sample_sizes = [20, 60, 100]
    ro_values = [0, 0.5, 0.9]
    iterations = 1000

    coef_types = ["Пирсон", "Спирмен", "Квадрантный"]
    storage = {
        ct: {dn: {size: {rho: [] for rho in ro_values} for size in sample_sizes} for dn in dist_names}
        for ct in coef_types
    }

    min_vals = {
        size: {rho: {'x': np.full(size, np.inf), 'y': np.full(size, np.inf)} for rho in ro_values + ["mix"]} 
        for size in sample_sizes
    }
    max_vals = {
        size: {rho: {'x': np.full(size, -np.inf), 'y': np.full(size, -np.inf)} for rho in ro_values + ["mix"]} 
        for size in sample_sizes
    }

    for i in range(iterations):
        for size in sample_sizes:
            for rho in ro_values:
                cov_matrix = [[1, rho], [rho, 1]]
                sample = random.multivariate_normal([0, 0], cov_matrix, size=size)
                x, y = sample[:, 0], sample[:, 1]
                
                min_vals[size][rho]['x'] = np.minimum(min_vals[size][rho]['x'], x)
                max_vals[size][rho]['x'] = np.maximum(max_vals[size][rho]['x'], x)
                
                min_vals[size][rho]['y'] = np.minimum(min_vals[size][rho]['y'], y)
                max_vals[size][rho]['y'] = np.maximum(max_vals[size][rho]['y'], y)

                storage["Пирсон"]["Нормальное"][size][rho].append(stats.pearsonr(x, y)[0])
                storage["Спирмен"]["Нормальное"][size][rho].append(stats.spearmanr(x, y)[0])
                storage["Квадрантный"]["Нормальное"][size][rho].append(calc_quadratic_correlation(x, y))

            cov_matrix1 = [[1, 0.9], [0.9, 1]]
            cov_matrix2 = [[10, -9], [-9, 10]]
            
            s1 = random.multivariate_normal([0, 0], cov_matrix1, size=size)
            s2 = random.multivariate_normal([0, 0], cov_matrix2, size=size)

            mixed_sample = s1 * 0.9 + s2 * 0.1
            mx, my = mixed_sample[:, 0], mixed_sample[:, 1]

            min_vals[size]["mix"]['x'] = np.minimum(min_vals[size]["mix"]['x'], mx)
            max_vals[size]["mix"]['x'] = np.maximum(max_vals[size]["mix"]['x'], mx)
                
            min_vals[size]["mix"]['y'] = np.minimum(min_vals[size]["mix"]['y'], my)
            max_vals[size]["mix"]['y'] = np.maximum(max_vals[size]["mix"]['y'], my)

            storage["Пирсон"]["Смешанное"][size][0].append(stats.pearsonr(mx, my)[0])
            storage["Спирмен"]["Смешанное"][size][0].append(stats.spearmanr(mx, my)[0])
            storage["Квадрантный"]["Смешанное"][size][0].append(calc_quadratic_correlation(mx, my))

    final_data = {ct: {"Среднее": {}, "Дисперсия": {}} for ct in coef_types}

    for ct in coef_types:
        for dn in dist_names:
            final_data[ct]["Среднее"][dn] = {size: {} for size in sample_sizes}
            final_data[ct]["Дисперсия"][dn] = {size: {} for size in sample_sizes}
            
            for size in sample_sizes:
                active_ros = ro_values if dn == "Нормальное" else [0]
                
                for rho in active_ros:
                    vals = storage[ct][dn][size][rho]
                    final_data[ct]["Среднее"][dn][size][rho] = np.mean(vals)
                    final_data[ct]["Дисперсия"][dn][size][rho] = np.var(vals)

    print_results_table(final_data, "Нормальное")
    print_results_table(final_data, "Смешанное")

    for size in sample_sizes:   
        plot_subplots_ellipses(
            size=size, 
            min_vals=min_vals, 
            max_vals=max_vals, 
            rho_keys=[0, 0.5, 0.9], 
            title_prefix="Нормальное распределение", 
            n_std=3.0 
        )
        
        plot_subplots_ellipses(
            size=size, 
            min_vals=min_vals, 
            max_vals=max_vals, 
            rho_keys=["mix"], 
            title_prefix="Смешанное распределение", 
            n_std=3.0 
        )

if __name__ == "__main__":
    main()