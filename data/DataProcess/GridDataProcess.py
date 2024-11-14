import numpy as np
from scipy.spatial import cKDTree

# 示例监测站数据
stations = {
    'lat': [39.65, 39.70, 39.80, 39.75],  # 纬度
    'lon': [118.15, 118.20, 118.10, 118.25],  # 经度
    'pm25': [70, 50, 40, 60]  # PM2.5 值
}

# 生成网格
grid_x, grid_y = np.meshgrid(
    np.linspace(39.60, 39.90, 50),  # 纬度范围
    np.linspace(118.05, 118.30, 50)  # 经度范围
)

# 计算IDW插值
def idw_interpolation(x, y, z, xi, yi, power=2):
    distances = np.sqrt((x[:, np.newaxis] - xi.ravel())**2 + (y[:, np.newaxis] - yi.ravel())**2)
    weights = 1 / (distances**power)
    weights[distances == 0] = 1e-10  # 防止除零
    z_interp = np.sum(weights * z[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    return z_interp.reshape(xi.shape)

# 应用IDW插值
interpolated_pm25 = idw_interpolation(
    np.array(stations['lat']),
    np.array(stations['lon']),
    np.array(stations['pm25']),
    grid_x, grid_y
)

# 可视化结果（可选）
import matplotlib.pyplot as plt
plt.contourf(grid_x, grid_y, interpolated_pm25, cmap='jet')
plt.colorbar(label='PM2.5')
plt.scatter(stations['lat'], stations['lon'], c='black', label='Stations')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('IDW Interpolation of PM2.5')
plt.legend()
plt.show()
