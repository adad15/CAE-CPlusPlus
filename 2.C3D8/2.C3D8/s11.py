import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

def read_displacement_data(filename):
    # 读取 Displacement_data.txt 文件
    data = np.loadtxt(filename)
    Nodes = data[:, :3]
    U = data[:, 3:]
    return Nodes, U

def read_stress_data(filename):
    # 读取 Stress_S11_data.txt 文件
    data = np.loadtxt(filename)
    S11 = data[:, 3]
    return S11

def read_mesh_data(filename, Nodes, tol=1e-4):
    # 读取 mesh_data.txt 文件，获取单元连接信息
    Elements = []
    with open(filename, 'r') as f:
        element_nodes = []
        for line in f:
            line = line.strip()
            if not line:
                if element_nodes:
                    Elements.append(element_nodes)
                    element_nodes = []
                continue
            coords = list(map(float, line.split()))
            element_nodes.append(coords)
        if element_nodes:
            Elements.append(element_nodes)

    Elements_indices = []
    for elem in Elements:
        if len(elem) != 8:
            raise ValueError(f"一个单元有 {len(elem)} 个节点，不是8个。请检查 mesh_data.txt 的格式。")
        indices = []
        for node in elem:
            # 使用近似匹配，避免浮点数精度问题
            distances = np.linalg.norm(Nodes - node, axis=1)
            matches = np.where(distances < tol)[0]
            if matches.size == 0:
                raise ValueError(f"节点坐标 {node} 在 Displacement_data.txt 中未找到匹配项。")
            elif matches.size > 1:
                raise ValueError(f"节点坐标 {node} 在 Displacement_data.txt 中有多个匹配项。")
            indices.append(matches[0])
        Elements_indices.append(indices)

    return np.array(Elements_indices)

def plot_contour(Nodes, Elements, U, Component, title="Component"):
    # 定义自定义颜色映射（蓝色到红色）
    myColor = np.array([
        [0, 0, 255],
        [0, 93, 255],
        [0, 185, 255],
        [0, 255, 232],
        [0, 255, 139],
        [0, 255, 46],
        [46, 255, 0],
        [139, 255, 0],
        [232, 255, 0],
        [255, 185, 0],
        [255, 93, 0],
        [255, 0, 0]
    ]) / 255.0

    cmap = ListedColormap(myColor)

    # 计算变形后的节点坐标
    deformation_coefficient = 5.0e2
    newNodes = Nodes + deformation_coefficient * U

    ElementNodeCount = Elements.shape[1]
    ElementCount = Elements.shape[0]

    # 定义单元类型（假设使用 C3D8 单元）
    cell_type = pv.CellType.HEXAHEDRON

    # 创建 PyVista 的 UnstructuredGrid
    cells = np.hstack([np.full((ElementCount, 1), ElementNodeCount), Elements]).astype(int)
    cells = cells.flatten()

    cell_types = np.full(ElementCount, cell_type, dtype=np.uint8)

    try:
        grid = pv.UnstructuredGrid(cells, cell_types, newNodes)
    except Exception as e:
        raise ValueError(f"创建网格时发生错误: {e}")

    # 检查网格是否正确创建
    print(f"网格节点数量: {grid.n_points}")
    print(f"网格单元数量: {grid.n_cells}")

    if grid.n_points == 0:
        raise ValueError("创建的网格没有任何节点。请检查输入数据是否正确。")
    if grid.n_cells == 0:
        raise ValueError("创建的网格没有任何单元。请检查输入数据是否正确。")

    # 确保 Component 的长度与节点数量一致
    if len(Component) != grid.n_points:
        raise ValueError(f"Component 的长度 ({len(Component)}) 与节点数量 ({grid.n_points}) 不一致。")

    # 将 Component 赋值为点数据（每个节点一个标量值）
    grid.point_data["Component"] = Component

    # 定义颜色映射的范围
    clim = [Component.min(), Component.max()]

    # 添加自定义颜色映射
    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="Component",
        cmap=cmap,
        show_edges=True,
        clim=clim,
        interpolate_before_map=True
    )

    # 设置视角为等角视图
    plotter.view_isometric()

    # 隐藏坐标轴
    plotter.hide_axes()

    # 添加颜色条（色标），移除不兼容的参数
    plotter.add_scalar_bar(
        title=title,
        n_labels=13,
        shadow=False
    )

    # 显示绘图窗口
    plotter.show()

if __name__ == "__main__":
    # 读取 Displacement_data.txt
    Nodes, U = read_displacement_data("Displacement_data.txt")

    # 读取 Stress_S11_data.txt
    S11 = read_stress_data("Stress_S11_data.txt")

    # 读取 mesh_data.txt 并获取单元连接信息
    Elements = read_mesh_data("mesh_data.txt", Nodes)

    # 检查 Component 长度是否与 Nodes 一致
    if len(S11) != len(Nodes):
        raise ValueError("Component 的长度与 Nodes 的数量不一致。")

    # 打印调试信息
    print(f"Nodes shape: {Nodes.shape}")
    print(f"Elements shape: {Elements.shape}")
    print(f"U shape: {U.shape}")
    print(f"S11 shape: {S11.shape}")

    # 绘制应力 S11 云图
    plot_contour(Nodes, Elements, U, S11, title="Stress S11")
