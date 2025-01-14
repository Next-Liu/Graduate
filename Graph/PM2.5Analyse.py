import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.matplotlib_fname())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据
labels = ['移动源', '生活源', '扬尘源', '工业源', '燃煤源']
sizes = [46, 16, 11, 10, 3]  # 各部分的大小
colors = ['orange', 'Pink', 'Green', 'Yellow', 'Cyan']  # 每部分的颜色
explode = (0.1, 0, 0, 0, 0)  # 突出显示第一个部分

# 绘制饼状图
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)

# 设置饼图的标题
plt.title('北京市PM2.5源分析')
plt.axis('equal')  # 确保饼图为圆形
plt.show()
