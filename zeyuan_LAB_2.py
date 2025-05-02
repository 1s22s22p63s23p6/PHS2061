import numpy as np
from math import sqrt, log10
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import monashspa.PHS3000 as spa
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.constants import c

background_per_second = 406/(13*60+59)
u_background_per_second = background_per_second * sqrt((sqrt(406)/406)**2 + (0.5/(13*60+59))**2)

count = [# 414,
        468,
        1259,
        847,
        1007,
        1922,
        1684,
        1896,
        1611,
        1323,
        1259,
        # 2407,
        1306,
        # 1894,
        1550,
        746,
        803,
        679,
        584,
        3389,
        2738,
        1755,
        3048,
        2455,
        1953,
        1508,
        1920,
        5187,
        286,
        556,
        1444,
        1435,
        1198,
        949,
        1829,
        1085,
        481,
        ]
# in seconds
time = [# 13*60+59,
        15*60+42,
        45*60+15,
        32*60+8,
        22*60+3,
        17*60+55,
        10*60+2,
        9*60+12,
        7*60+8,
        5*60+59,
        6*60+0,
        # 11*60+0,
        7*60+41,
        # 9*60+11,
        14*60+34,
        4*60+55,
        13*60+7,
        15*60+29,
        14*60+30,
        15*60+56,
        11*60+12,
        7*60+11,
        10*60+35,
        8*60+48,
        7*60+1,
        5*60+27,
        6*60+52,
        17*60+32,
        5*60+12,
        7*60+49,
        11*60+51,
        6*60+55,
        13*60+25,
        4*60+45,
        14*60+4,
        17*60+8,
        3*60+40,
        ]
        
u_time = 0.5
# in Amperes
current = [# 0,
           0.095,
           0.200,
           0.305,
           0.405,
           0.500,
           0.600,
           0.700,
           0.801,
           0.902,
           1.006,
           # 1.100,
           1.203,
           # 1.303,
           1.402,
           1.310,
           1.510,
           1.610,
           1.710,
           1.810,
           1.884,
           1.821,
           1.835,
           1.8365,
           1.841,
           1.842,
           1.836,
           1.855,
           1.750,
           1.760,
           1.782,
           1.805,
           1.452,
           1.102,
           1.353,
           0.450,
           0.530,
        ]
u_current = [# 0.00001,
             0.0002,
             0.001,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             #0.002,
             0.002,
             # 0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.006,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             0.002,
             ]

u_n = []
ns = []
for n in range(100,10000):
    u_n += [sqrt(n)/n]
    ns += [n]

if False:
    plt.plot(ns, u_n)
    plt.xlabel('n')
    plt.ylabel('u_n [%]')
    plt.title('u_n vs n')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.axhline(y=0.02, color='r', linestyle='--')
    plt.grid()
    plt.show()

count = np.array(count)
time = np.array(time)
u_current = np.array(u_current)
current = np.array(current)

# 计算计数率及其不确定度
counts_per_second = count / time
# 计数率的不确定度应该考虑计数的不确定度和时间的不确定度
u_counts_per_second = counts_per_second * np.sqrt((np.sqrt(count)/count)**2 + (u_time/time)**2)

# 校正背景后的计数率及其不确定度
counts_per_second_clean = counts_per_second - background_per_second
u_counts_per_second_clean = np.sqrt(u_counts_per_second**2 + u_background_per_second**2)

plt.figure(figsize=(10, 6))

# 使用大小来表示时间（将时间归一化为合适的点大小范围）
size_scale = 300  # 调整这个值以获得合适的点大小
sizes = time / np.max(time) * size_scale + 1  # 加30确保最小点也可见

# 创建散点图，颜色代表计数，大小代表时间
sc = plt.scatter(current, 
                 counts_per_second_clean, 
                 c=np.log10(count),  # 颜色映射到计数
                 s=sizes,  # 大小映射到时间
                 cmap='viridis', 
                 alpha=0.7,  # 稍微透明以便看清重叠点
                 zorder=2,
                 label='Data points')

plt.errorbar(current,
             counts_per_second_clean,
             xerr=u_current,
             yerr=u_counts_per_second_clean,
             fmt='none', 
             capsize=1,
             ecolor='black',
             zorder=1) 

# 添加颜色条标注计数
cbar = plt.colorbar(sc)
cbar.set_label('Counts(log10 scale)')

# 添加时间图例
# 创建一些虚拟点来表示不同的时间
time_legend_values = [np.min(time), np.mean(time), np.max(time)]
time_legend_sizes = [t / np.max(time) * size_scale + 30 for t in time_legend_values]
time_legend_labels = [f'{int(t/60)}m {int(t%60)}s' for t in time_legend_values]

# 添加时间图例
for i, (size, label) in enumerate(zip(time_legend_sizes, time_legend_labels)):
    plt.scatter([], [], s=size, c='gray', alpha=0.7, label=f'Time: {label}')

plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('Current (A)')
plt.ylabel('Counts per second (background subtracted)')
plt.title('Counts per second vs Current')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', title='Time Legend')
plt.show()

# 高斯拟合部分 - 针对高电流值(>1.75A)
# 定义高斯函数
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# 筛选电流值大于1.75的数据
high_current_mask = current >= 1.75
high_current = current[high_current_mask]
high_counts_ps = counts_per_second_clean[high_current_mask]
high_u_current = u_current[high_current_mask]
high_u_counts_ps = u_counts_per_second_clean[high_current_mask]

# 如果有足够的数据点，执行高斯拟合
if len(high_current) >= 3:
    # 进行高斯拟合
    initial_guess = [np.max(high_counts_ps), np.mean(high_current), 0.05]
    try:
        popt, pcov = curve_fit(gaussian, high_current, high_counts_ps, p0=initial_guess, 
                              sigma=high_u_counts_ps, absolute_sigma=True)
        
        # 提取拟合参数和不确定度
        A, mu, sigma = popt
        u_A, u_mu, u_sigma = np.sqrt(np.diag(pcov))
        
        # 生成拟合曲线的x和y值
        x_fit = np.linspace(min(high_current)-0.05, max(high_current)+0.05, 1000)
        y_fit = gaussian(x_fit, A, mu, sigma)
        
        # 绘制原始数据点和拟合曲线
        plt.figure(figsize=(10, 6))
        plt.errorbar(high_current, high_counts_ps, xerr=high_u_current, yerr=high_u_counts_ps, 
                    fmt='o', label='Data points', capsize=3, ecolor='black')
        plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')
        
        plt.xlabel('Current (A)')
        plt.ylabel('Counts per second (background subtracted)')
        plt.title('Gaussian Fit for Current > 1.75A')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示拟合参数
        fit_info = f'Fit parameters:\nA = {A:.2f} ± {u_A:.2f}\nμ = {mu:.4f} ± {u_mu:.4f}\nσ = {sigma:.4f} ± {u_sigma:.4f}'
        plt.annotate(fit_info, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # 添加一倍不确定度范围曲线
        # 计算高斯函数的上下界
        y_upper = gaussian(x_fit, A + u_A, mu - u_mu, sigma - u_sigma)  # 最大值情况
        y_lower = gaussian(x_fit, A - u_A, mu + u_mu, sigma + u_sigma)  # 最小值情况
        
        # 绘制不确定度区域
        plt.fill_between(x_fit, y_lower, y_upper, color='r', alpha=0.2, label='1σ uncertainty')
        
        plt.legend()
        plt.show()
        
        print("\n--- 高斯拟合结果 ---")
        print(f"振幅 A = {A:.4f} ± {u_A:.4f}")
        print(f"中心值 μ = {mu:.4f} ± {u_mu:.4f}")
        print(f"标准差 σ = {sigma:.4f} ± {u_sigma:.4f}")
        print(f"全宽半高 FWHM = {2.355 * sigma:.4f} ± {2.355 * u_sigma:.4f}")
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
else:
    print("没有足够的高电流数据点进行高斯拟合(需要至少3个点)")

print(u_counts_per_second)

print(u_counts_per_second_clean)

print(u_background_per_second)

# 假定你已经得到高斯拟合均值 mu_g（单位 A），以及校准常数 k_conv 已用前面的代码计算（例如 k_conv ≈ 0.345 MeV/A）
# 注意：这里示例中 mu_g 和 k_conv 请按实际拟合结果替换
k_conv = 0.9997 / mu  # 根据661.7 keV（0.62421 MeV）动能与电子静止能转换得到的动量计算，
                         # p = sqrt((T + m_e)^2 - m_e^2), 此处校准常数即 p_mu/mu_g
print(f"校准常数 k_conv = {k_conv:.5f} MeV/A")

# 将现有电流转换为粒子的动量``
# current 数组已在之前代码中定义
momentum = k_conv * current  # 单位 MeV

# 根据提示，由于 Δp ∝ p，因此要消除这种效应，将背景扣除后的计数率除以 p
# 注意：要避免 p = 0 的情况，因此先筛选 current > 0
mask = current > 0
p_nonzero = momentum[mask]
counts_nonzero = counts_per_second_clean[mask]
u_counts_nonzero = u_counts_per_second_clean[mask]
u_current_nonzero = u_current[mask]  # 电流的不确定度

# 计算校正后的谱：每单位动量内的计数率
corrected_counts = counts_nonzero / p_nonzero
u_corrected_counts = u_counts_nonzero / p_nonzero

# 绘图展示
plt.figure(figsize=(10, 6))
# 这里xerr 根据校准常数将 u_current 转换为动量不确定度
plt.errorbar(p_nonzero, corrected_counts, xerr=k_conv * u_current_nonzero,
             yerr=u_corrected_counts, fmt='o', capsize=3, ecolor='black',
             label='Corrected Counts per Unit Momentum')
plt.xlabel("Momentum p (MeV)")
plt.ylabel("Counts per unit momentum (arb. units)")
plt.title("Corrected Counts per Unit Momentum vs Momentum")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# 假定前面的代码已经得到如下变量：
# p_nonzero：对应非零电流转换得到的动量数据（单位 MeV）
# corrected_counts：每单位动量内的计数率
# u_corrected_counts：计数率的不确定度
# u_current_nonzero：原始电流不确定度，用于换算成动量不确定度（xerr），已通过 k_conv 转换

# 例如我们希望在动量范围 [p_min, p_max] 内进行拟合
p_min = 0.105  # MeV，示例：下界
p_max = 0.55  # MeV，示例：上界

# 筛选出这一范围内的数据
fit_mask = (p_nonzero >= p_min) & (p_nonzero <= p_max)
p_fit_data = p_nonzero[fit_mask]
counts_fit_data = corrected_counts[fit_mask]
u_counts_fit_data = u_corrected_counts[fit_mask]
u_current_fit_data = u_current_nonzero[fit_mask]  # 用于计算 xerr


import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# 基础物理常量
m_e = 0.511  # 电子静止能量 (MeV)

# 库仑修正因子
def fermi_function(Z, W, p):
    """库仑修正因子 F(Z, W)
    
    参数:
        Z: 原子序数
        W: 总能量 (MeV)
        p: 动量 (MeV/c)
    """
    alpha = 1 / 137  # 精细结构常数
    eta = Z * alpha * W / p
    # 避免除零错误
    if np.any(p == 0):
        return 1.0
    return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))

# 形状因子
def shape_factor(p, W, lambd):
    """β衰变形状因子
    
    参数:
        p: 动量 (MeV/c)
        W: 总能量 (MeV)
        lambd: 形状参数
    """
    # 简化形状因子计算，使用二阶多项式
    a0 = 1.0 + 0.02 * lambd * W
    a1 = -0.04 * lambd * p
    a2 = 0.01 * lambd**2 * p**2
    return a0 + a1 * p + a2 * p**2

# β衰变动量谱模型
def beta_spectrum(p, C, W0, Z, lambd, x_shift=0):
    """β衰变动量谱理论模型
    
    参数:
        p: 动量数组 (MeV/c)
        C: 归一化常数
        W0: 端点能量 (MeV)
        Z: 原子序数
        lambd: 形状参数
        x_shift: 动量平移参数 (MeV/c)
    
    返回:
        每单位动量的β粒子数
    """
    # 平移动量
    p_shifted = p + x_shift
    
    # 计算总能量
    W = np.sqrt(p_shifted**2 + m_e**2)
    
    # 初始化全零数组
    result = np.zeros_like(p)
    
    # 只在物理允许区域(W < W0)计算
    mask = W < W0
    
    if not np.any(mask):
        return result
    
    # 计算库仑修正因子
    F = fermi_function(Z, W[mask], p_shifted[mask])
    
    # 计算形状因子
    S = shape_factor(p_shifted[mask], W[mask], lambd)
    
    # 计算动量谱
    # 注意(W0-W)^2项确保在W=W0处为零
    result[mask] = C * (p_shifted[mask]**2 / W[mask]) * (W0 - W[mask])**2 * F * S
    
    return result

# 拟合主程序
def fit_beta_spectrum(p_data, counts_data, u_counts, p_min=0.1, p_max=0.7, fixed_W0=None):
    """拟合β衰变动量谱"""
    # 筛选拟合范围内的数据
    fit_mask = (p_data >= p_min) & (p_data <= p_max)
    p_fit = p_data[fit_mask]
    counts_fit = counts_data[fit_mask]
    u_counts_fit = u_counts[fit_mask]
    
    print(f"拟合数据点数: {len(p_fit)}")
    
    # 参数边界
    if fixed_W0 is not None:
        bounds = [
            (0, 1E10),      # C: 归一化常数
            (fixed_W0, fixed_W0),  # W0: 固定端点能量
            (50, 60),       # Z: 原子序数
            (-2, 2),        # lambd: 形状参数
            (-0.1, 0.1)     # x_shift: 动量平移参数
        ]
    else:
        bounds = [
            (0, 1E10),      # C: 归一化常数
            (0.7, 1.2),     # W0: 端点能量范围
            (50, 60),       # Z: 原子序数
            (-2, 2),        # lambd: 形状参数
            (-0.1, 0.1)     # x_shift: 动量平移参数
        ]
    
    # 残差函数
    def residuals(params):
        C, W0, Z, lambd, x_shift = params
        model = beta_spectrum(p_fit, C, W0, Z, lambd, x_shift)
        residual = (model - counts_fit) / u_counts_fit
        return np.sum(residual**2)
    
    # 全局优化
    print("开始全局优化...")
    result = differential_evolution(residuals, bounds, maxiter=3000, tol=1e-6)

# 执行拟合
print("执行β谱拟合...")
fit_mask = (p_nonzero >= p_min) & (p_nonzero <= p_max)
p_fit = p_nonzero[fit_mask]
counts_fit = corrected_counts[fit_mask]
u_counts_fit = u_corrected_counts[fit_mask]

# 残差函数
def residuals(params):
    C, W0, Z, lambd, x_shift = params
    model = beta_spectrum(p_fit, C, W0, Z, lambd, x_shift)
    residual = (model - counts_fit) / u_counts_fit
    return np.sum(residual**2)

# 参数边界
bounds = [
    (0, 1E10),      # C: 归一化常数
    (0.7, 1.2),     # W0: 端点能量范围
    (50, 60),       # Z: 原子序数
    (-2, 2),        # lambd: 形状参数
    (-0.1, 0.1)     # x_shift: 动量平移参数
]

# 全局优化
print("开始全局优化...")
result = differential_evolution(residuals, bounds, maxiter=3000, tol=1e-6)

# 提取拟合参数
popt = result.x
C_fit, W0_fit, Z_fit, lambd_fit, x_shift_fit = popt

# 计算临界动量 (W0对应的动量)
p_critical = np.sqrt(W0_fit**2 - m_e**2)

print("\n=== 拟合结果 ===")
print(f"C      = {C_fit:.5e}")
print(f"W0     = {W0_fit:.5f} MeV")
print(f"Z      = {Z_fit:.2f}")
print(f"lambd  = {lambd_fit:.5f}")
print(f"x_shift = {x_shift_fit:.5f} MeV/c")
print(f"临界动量 p_critical = {p_critical:.5f} MeV/c (W={W0_fit:.5f} MeV)")

# 创建扩展的动量范围用于绘图，确保包含临界点
p_extended = np.linspace(min(p_fit), p_critical*1.1, 1000)
model_extended = beta_spectrum(p_extended, C_fit, W0_fit, Z_fit, lambd_fit, x_shift_fit)

# 计算拟合值和残差
fitted_values = beta_spectrum(p_fit, C_fit, W0_fit, Z_fit, lambd_fit, x_shift_fit)
normalized_residuals = (counts_fit - fitted_values) / u_counts_fit

# 绘制拟合图
plt.figure(figsize=(12, 8))
plt.errorbar(p_nonzero, corrected_counts, xerr=k_conv*u_current_nonzero,
             yerr=u_corrected_counts, fmt='o', capsize=3, ecolor='black',
             label='数据点', alpha=0.7)
plt.plot(p_extended, model_extended, 'r-', lw=2, label='拟合模型')
plt.axvline(x=p_critical, color='k', linestyle='--', 
            label=f'临界动量: p={p_critical:.4f} MeV/c')

# 在图上显示拟合参数
textstr = '\n'.join([
    f'W0 = {W0_fit:.4f} MeV',
    f'Z = {Z_fit:.2f}',
    f'λ = {lambd_fit:.4f}',
    f'x_shift = {x_shift_fit:.4f} MeV/c'
])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=props)

plt.xlabel("动量 p (MeV/c)")
plt.ylabel("每单位动量计数 (任意单位)")
plt.title("β衰变动量谱拟合")
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制残差图
plt.figure(figsize=(12, 5))
plt.scatter(p_fit, normalized_residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-')
plt.axhline(y=1, color='k', linestyle='--')
plt.axhline(y=-1, color='k', linestyle='--')
plt.grid(True, alpha=0.3)
plt.xlabel("动量 p (MeV/c)")
plt.ylabel("标准化残差 (σ)")
plt.title("拟合残差")

plt.tight_layout()
plt.show()


fermi_data = spa.betaray.modified_fermi_function_data
interpolated_fermi_data = interp1d(fermi_data[:,0],fermi_data[:,1],kind="cubic")
my_G = interpolated_fermi_data(p_nonzero[1:])


'''
使用实验提供的修正费米函数数据。
+ m_e**2
'''



W= np.sqrt((p_nonzero[1:]**2)+(m_e**2))
w = W 
corrected_counts_for_sn = corrected_counts[1:]

fit_mask_for_sn = (w >= 0.65) & (w <= 1)

slope_for_sn , intercept_for_sn, *_ =linregress(w[fit_mask_for_sn], corrected_counts_for_sn[fit_mask_for_sn])
w0 =  -intercept_for_sn/slope_for_sn
# 7. 绘图
plt.figure(figsize=(10,6))
plt.plot(w, slope_for_sn * w + intercept_for_sn, 'r--', label='Linear fit')
plt.errorbar(w, corrected_counts_for_sn, yerr=0, fmt='o', label='counts plot', capsize=3)
plt.axvline(x=w0, linestyle='--', color='k', label=f'w₀ = {w0:.4f}')
plt.axvline(x=0.65, color='r', linestyle='--', label='w = 0.39')
plt.axvline(x=1, color='g', linestyle='--', label='w = 1.0')
plt.xlabel('w')
plt.ylabel('counts')
plt.title('counts Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

print(f"Kurie图线性拟合结果：")
print(f"斜率 slope = {slope_for_sn:.5f}")
print(f"截距 intercept = {intercept_for_sn:.5f}")
print(f"端点能量 w₀ = {w0:.5f}")
print(f"换算为 MeV: W₀ = {w0 * m_e:.5f} MeV")




Sn = w**2 - 0.25 * (w0 - w)**2


#w = np.sqrt((p_nonzero[1:]**2)+ (m_e**2))  # MeV
  # dimensionless


kurie_y = np.sqrt(corrected_counts[1:] / (p_nonzero[1:] * w * my_G *Sn))
# 4. Error in Kurie variable
u_kurie_y = 0.5 * kurie_y * (u_corrected_counts[1:] / corrected_counts[1:])

low = 0.72
up = 1

fit_mask = (w >= low) & (w <= up)

slope, intercept, *_ = linregress(w[fit_mask], kurie_y[fit_mask])

# 6. 计算 w₀（x轴截距）
w0_kurie = -intercept / slope



def linear_model(x, a, b):
    return a * x + b
from scipy.optimize import curve_fit

popt, pcov = curve_fit(
    linear_model,
    w[fit_mask], kurie_y[fit_mask],               # 自变量与因变量
    sigma=u_kurie_y[fit_mask],                    # 每个点的y误差
    absolute_sigma=True                           # 告诉curve_fit使用真实误差（不是相对权重）
)

slope, intercept = popt
slope_err, intercept_err = np.sqrt(np.diag(pcov))

plot_mask = (w >= low) & (w <= 114514)

w_fit = np.linspace(min(w[plot_mask]), max(w[plot_mask]), 300)
y_fit = linear_model(w_fit, slope, intercept)
# 拟合曲线每个点的一倍σ：误差传播公式
y_uncertainty = np.sqrt((w_fit * slope_err)**2 + intercept_err**2)




# 7. 绘图
plt.figure(figsize=(10,6))
plt.errorbar(w[fit_mask], kurie_y[fit_mask],yerr=u_kurie_y[fit_mask], fmt='o', label='Corrected Kurie plot', capsize=3)
plt.plot(w[plot_mask], slope * w[plot_mask] + intercept, label='Linear fit' ,color='red')
plt.axhline(y = 0, color='k', label=f'y =0')
plt.fill_between(w_fit, y_fit - y_uncertainty, y_fit + y_uncertainty,
                 color='red', alpha=0.2, label='1σ uncertainty band')

fit_label = (    
    f"Fit: $y = ax + b$\n"
    f"$a$ = {slope:.2f} ± {slope_err:.2f}\n"
    f"$b$ = {intercept:.2f} ± {intercept_err:.2f}\n"
    f"$w_0$ = {w0_kurie:.4f}"
    )
plt.plot([], [], ' ', label=fit_label)
plt.xlabel('w')
plt.ylabel('Kurie variable')
plt.title('Kurie Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
plt.savefig("figure.svg", format="svg")
print(f"Kurie图线性拟合结果：")
print(f"斜率 slope = {slope:.5f}")
print(f"截距 intercept = {intercept:.5f}")
print(f"端点能量 w₀ = {w0_kurie:.5f}")
print(f"换算为 MeV: W₀ = {w0_kurie * m_e:.5f} MeV")

