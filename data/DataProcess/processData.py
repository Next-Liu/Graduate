import numpy as np
import matplotlib.pyplot as plt

# 经纬度范围
lon_start, lon_end = 118.13, 118.22
lat_start, lat_end = 39.61, 39.68

# 网格间隔
grid_step = 0.01

# 生成经纬度网格
lons = np.arange(lon_start, lon_end,grid_step)
lats = np.arange(lon_start, lon_end,grid_step)

# 计算网格点总数
grid_points = len(lons) * len(lats)
print(f"Total grid points: {grid_points}")

# 创建网格点的坐标

lon_grid, lat_grid = np.meshgrid(lons, lats)

# 绘制网格点
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(lon_grid, lat_grid, color='blue', s=1)  # 使用散点绘制网格点，s 是点大小

# 将指定的点标注出来
highlight_lons = [118.13, 118.17, 118.22, 118.19]
highlight_lats = [39.64, 39.63, 39.67, 39.62]
ax.scatter(highlight_lons, highlight_lats, color='red', marker='o', label="Highlighted Points")

# 添加标签
ax.set_title("Region Grid with Highlighted Points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()
