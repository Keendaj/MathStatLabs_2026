import numpy as np
from scipy import stats

def calc_confidence_intervals(sample, alpha=0.05):
    n = len(sample)
    
    mean_hat = np.mean(sample)
    var_hat = np.var(sample, ddof=1)
    std_hat = np.std(sample, ddof=1)
    
    u_crit = stats.norm.ppf(1 - alpha/2)
    
    margin_of_error = u_crit * (std_hat / np.sqrt(n))
    ci_mean = (mean_hat - margin_of_error, mean_hat + margin_of_error)
    
    e = stats.kurtosis(sample, fisher=True)

    U = u_crit * np.sqrt((e + 2) / n)
    
    lower_std = std_hat * (1 + U)**(-0.5)
    upper_std = std_hat * (1 - U)**(-0.5) if U < 1 else np.inf 
    
    ci_var = (lower_std**2, upper_std**2)
    
    return mean_hat, var_hat, ci_mean, ci_var

def f_test_methodical(sample1, sample2, alpha=0.05):
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    n1 = len(sample1)
    n2 = len(sample2)
    
    if var1 >= var2:
        var_max, var_min = var1, var2
        df_num, df_den = n1 - 1, n2 - 1
        num_name = "Выборка 1"
    else:
        var_max, var_min = var2, var1
        df_num, df_den = n2 - 1, n1 - 1
        num_name = "Выборка 2"
        
    f_stat = var_max / var_min
    
    f_crit = stats.f.ppf(1 - alpha, df_num, df_den)
    
    if f_stat > f_crit:
        decision = f"ОТКЛОНЯЕТСЯ (Дисперсия '{num_name}' значимо больше)"
    else:
        decision = "Принимается (H0: Дисперсии равны)"
        
    return f_stat, f_crit, decision, num_name

def main():
    rng = np.random.default_rng(4)
    alpha = 0.05
    
    n1, n2 = 20, 100
    sample1 = rng.normal(loc=0, scale=1, size=n1)
    sample2 = rng.normal(loc=0, scale=1, size=n2)
    
    samples = [
        (f"Выборка 1 (n={n1})", sample1),
        (f"Выборка 2 (n={n2})", sample2)
    ]
    
    print("="*95)
    print("2. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (α = 0.05)")
    print("="*95)
    print(f"{'Выборка':<18} | {'μ^ (Среднее)':<12} | {'ДИ для мат. ожидания':<25} | {'s² (Дисперсия)':<14} | {'ДИ для дисперсии':<25}")
    print("-" * 95)
    
    for name, sample in samples:
        mean, var, ci_mean, ci_var = calc_confidence_intervals(sample, alpha)
        ci_mean_str = f"[{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]"
        ci_var_str = f"[{ci_var[0]:.3f}, {ci_var[1]:.3f}]"
        
        print(f"{name:<18} | {mean:>12.3f} | {ci_mean_str:<25} | {var:>14.3f} | {ci_var_str:<25}")

    print("\n" + "="*95)
    print("3. F-ТЕСТ (КРИТЕРИЙ ФИШЕРА) НА РАВЕНСТВО ДИСПЕРСИЙ (α = 0.05)")
    print("="*95)
    
    f_stat, f_crit, decision, num_name = f_test_methodical(sample1, sample2, alpha)
    
    print("H0: Дисперсии равны (σ1² = σ2²)")
    print(f"H'1: Дисперсия бóльшей выборки > Дисперсии меньшей")
    print("-" * 95)
    print(f"В числителе          : {num_name} (дисперсия больше)")
    print(f"F-статистика (набл.) : {f_stat:.4f}")
    print(f"F-критическое (прав.): {f_crit:.4f}")
    print(f"Условие F > F_crit   : {f_stat > f_crit}")
    print(f"Результат            : {decision}")
    print("="*95)

if __name__ == "__main__":
    main()