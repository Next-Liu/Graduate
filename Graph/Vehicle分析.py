import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.matplotlib_fname())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据
labels = ['汽车', '低速汽车', '摩托车', '挂车']
sizes = [90.1, 2.2, 4.6, 3.1]  # 各部分的大小
colors = ['deepskyblue', 'orange', 'Green', 'Yellow']  # 每部分的颜色

# 绘制饼状图
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)

# 设置饼图的标题
plt.title('NOX')
plt.axis('equal')  # 确保饼图为圆形
plt.show()
