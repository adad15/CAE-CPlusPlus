import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 读取数据
node = np.loadtxt('node.csv', delimiter=',')
new_node = np.loadtxt('new_node.csv', delimiter=',')
element = np.loadtxt('element.csv', delimiter=',', dtype=int) - 1  # 减 1，因为 Python 索引从 0 开始
sgm = np.loadtxt('sgm.csv')

# 绘制原始形状和变形后的形状
plt.figure()
for elem in element:
    polygon = node[elem]
    plt.fill(*zip(*polygon), edgecolor='r', fill=False)
for elem in element:
    polygon = new_node[elem]
    plt.fill(*zip(*polygon), edgecolor='b', fill=False)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Deformation')
plt.legend(['Original', 'Deformed'])
plt.axis('equal')
plt.show()

# 绘制应力分布（σ11）
plt.figure()
triang = tri.Triangulation(node[:,0], node[:,1], triangles=element)
plt.tripcolor(triang, facecolors=sgm, cmap='jet', edgecolors='k')
plt.colorbar(label=r'$\sigma_{11}$ (Pa)')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'$\sigma_{11}$')
plt.axis('equal')
plt.show()
