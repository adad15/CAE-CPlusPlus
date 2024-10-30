#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

// 定义一些全局变量
double E = 210e9; // 弹性模量，单位 Pa
double u = 0.4;   // 泊松比，无量纲
double q = 100;   // 外载荷（面力），单位 N/㎡
double t = 0.1;   // 厚度，单位 m
double bt = (3 + 0.5407762) * 1e11; // 乘大数法的大数，正常做法把边界条件对应的刚度矩阵中的行列删除，但是这样做结点标号会乱，和原本标号对不上，所以使用乘大数法
double length = 5.0; // 长度，单位 m
double height = 1.0; // 高度，单位 m
// 表示二维向量，其 x 分量和 y 分量都被设置为 0.2
Vector2d ele_size(0.2, 0.2); // 单元的两个直角边尺寸

// 应力应变矩阵 D
Matrix3d D;

// 计算单元刚度矩阵的函数
// element：元素连接矩阵，存储每个单元对应的节点索引
// node：节点坐标向量，每个节点是一个二维向量Vector2d
// ele_id：待计算的单元编号（假设从 0 开始计数）
MatrixXd ele_stiff_matrix(const vector<vector<int>>& element, const vector<Vector2d>& node, int ele_id, double A, const Matrix3d& D, double t) {
	int i = ele_id; // 假设 ele_id 从 0 开始计数

	// 获取第i个单元的节点索引
	int n1 = element[i][0]; // 节点 1
	int n2 = element[i][1]; // 节点 2
	int n3 = element[i][2]; // 节点 3

	// 获取节点坐标
	double x1 = node[n1](0);
	double y1 = node[n1](1);
	double x2 = node[n2](0);
	double y2 = node[n2](1);
	double x3 = node[n3](0);
	double y3 = node[n3](1);

	cout << "Element " << i << " nodes: (" << x1 << "," << y1 << "), ("
		<< x2 << "," << y2 << "), (" << x3 << "," << y3 << ")" << endl;

	// 计算系数
	double a1 = x2 * y3 - x3 * y2;
	double a2 = x3 * y1 - x1 * y3;
	double a3 = x1 * y2 - x2 * y1;

	double b1 = y2 - y3;
	double b2 = y3 - y1;
	double b3 = y1 - y2;

	double c1 = -x2 + x3;
	double c2 = -x3 + x1;
	double c3 = -x1 + x2;

	// 计算应变-位移矩阵 B
	double factor = 1.0 / (2.0 * A);

	Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor; //B 阵为常数阵

	// 计算单元刚度矩阵 ke
	MatrixXd ke = B.transpose() * D * B * t * A;

	cout << "Element " << i << " stiffness matrix ke:" << endl;
	cout << ke << endl;

	return ke; // 6*6的矩阵，3*2（3个结点，每个结点2个自由度）
}

// 计算单元应力的函数
Vector3d stress_calculate(const vector<vector<int>>& element, const vector<Vector2d>& node, int ele_id, double A, const Matrix3d& D, double t, const VectorXd& a) {
	int i = ele_id;

	// 获取第 i 个单元的节点索引
	int n1 = element[i][0];
	int n2 = element[i][1];
	int n3 = element[i][2];

	// 获取节点坐标
	double x1 = node[n1](0);
	double y1 = node[n1](1);
	double x2 = node[n2](0);
	double y2 = node[n2](1);
	double x3 = node[n3](0);
	double y3 = node[n3](1);

	// 计算系数
	double a1 = x2 * y3 - x3 * y2;
	double a2 = x3 * y1 - x1 * y3;
	double a3 = x1 * y2 - x2 * y1;

	double b1 = y2 - y3;
	double b2 = y3 - y1;
	double b3 = y1 - y2;

	double c1 = -x2 + x3;
	double c2 = -x3 + x1;
	double c3 = -x1 + x2;

	// 计算应变-位移矩阵 B
	double factor = 1.0 / (2.0 * A);

	Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor;

	// 提取节点位移 ae
	VectorXd ae(6);
	ae << a(2 * n1), a(2 * n1 + 1),
		a(2 * n2), a(2 * n2 + 1),
		a(2 * n3), a(2 * n3 + 1);

	// 计算单元应变
	Vector3d yps = B * ae;

	// 计算单元应力
	Vector3d sgm = D * yps;

	return sgm;
}

int main() {
	// 初始化 D 矩阵
	D << E / (1 - u * u), u* E / (1 - u * u), 0,
		u* E / (1 - u * u), E / (1 - u * u), 0,
		0, 0, E / (2 * (1 + u));

	int ele_H = static_cast<int>(length / ele_size(0)); // 水平单元数
	int ele_V = static_cast<int>(height / ele_size(1)); // 竖向单元数
	int ele_num = ele_H * ele_V * 2; // 单元数
	int node_num = (ele_H + 1) * (ele_V + 1); // 节点数,156
	int DOF = 2; // 每个节点自由度

	// 定义每个单元的节点号
	vector<vector<int>> element(ele_num, vector<int>(3));
	int num = 0;
	for (int i = 0; i < ele_H; ++i) {
		for (int j = 0; j < ele_V; ++j) {
			// 下三角，逆时针编号
			element[num][0] = j + i * (ele_V + 1);
			element[num][1] = j + (i + 1) * (ele_V + 1);
			element[num][2] = j + 1 + i * (ele_V + 1);
			num++;
			element[num][0] = j + 1 + i * (ele_V + 1);
			element[num][1] = j + (i + 1) * (ele_V + 1);
			element[num][2] = j + 1 + (i + 1) * (ele_V + 1);
			num++;
		}
	}

	for (int i = 0; i < num; i++) {
		cout << element[i][0] << " " << element[i][1] << " " << element[i][2] << endl;;
	}

	// 定义节点坐标，使用 vector<Vector2d>
	vector<Vector2d> node;
	for (int i = 0; i <= ele_H; ++i) {
		for (int j = 0; j <= ele_V; ++j) {
			Vector2d coord;
			coord(0) = i * ele_size(0); // x 坐标
			coord(1) = j * ele_size(1); // y 坐标
			node.push_back(coord);
		}
	}
	cout << "----------------------------------------" << endl;
	for (int i = 0; i < node.size(); i++) {
		cout << node[i](0) << " " << node[i](1) << endl;
	}

	double A = 0.5 * ele_size(0) * ele_size(1); // 单元面积

	// 每个单元自由度在整体矩阵中的索引位置
	vector<vector<int>> ele_DOF(ele_num, vector<int>(DOF * 3)/*单刚维度*/);
	for (int i = 0; i < ele_num; ++i) {
		for (int j = 0; j < 3; ++j) { // 三角形三个节点循环,逆时针
			int node_id = element[i][j];
			ele_DOF[i][2 * j] = 2 * node_id;
			ele_DOF[i][2 * j + 1] = 2 * node_id + 1;
		}
	}

	for (int i = 0; i < ele_num; ++i) {
		for (int j = 0; j < 6; j++) {
			cout << ele_DOF[i][0] << " " << ele_DOF[i][1] << " " << ele_DOF[i][2] << " " << ele_DOF[i][3] << " " << ele_DOF[i][4] << " "
				<< ele_DOF[i][5] << endl;
		}
	}

	// 组装全局刚度矩阵
	int total_DOF = node_num * DOF;
	MatrixXd KZ = MatrixXd::Zero(total_DOF, total_DOF); //维度156*2=312

	for (int i = 0; i < ele_num; ++i) {
		MatrixXd ke = ele_stiff_matrix(element, node, i, A, D, t); // 第i个单元的刚度矩阵
		for (int m = 0; m < ke.rows(); ++m) {
			int row = ele_DOF[i][m];
			for (int n = 0; n < ke.cols(); ++n) {
				int col = ele_DOF[i][n];
				KZ(row, col) += ke(m, n);
			}
		}
	}

	ofstream KZ_file("KZ_matrix.csv");
	if (KZ_file.is_open()) {
		for (int i = 0; i < KZ.rows(); ++i) {
			for (int j = 0; j < KZ.cols(); ++j) {
				KZ_file << KZ(i, j);
				if (j < KZ.cols() - 1) {
					KZ_file << ","; // 逗号分隔
				}
			}
			KZ_file << endl;
		}
		KZ_file.close();
		cout << "全局刚度矩阵 KZ 已保存到文件 KZ_matrix.csv" << endl;
	}
	else {
		cout << "无法打开文件 KZ_matrix.csv 进行写入。" << endl;
	}

	// 定义载荷向量 PZ，维度312
	VectorXd PZ = VectorXd::Zero(total_DOF);

	// 施加外载荷（需要根据具体的索引调整）
	PZ(2 * ele_V + 1) = -0.5 * (q * t * length) / ele_H;
	for (int i = 1; i < ele_H; ++i) {
		int idx = 11 + 2 * (ele_V + 1) * i;
		if (idx < total_DOF) {
			PZ(idx) = -q * t * length / ele_H;
		}
	}
	PZ(total_DOF - 1) = -0.5 * q * t * 5 / 10;

	for (int i = 0; i < total_DOF; i++) cout << PZ(i) << endl;

	// 施加约束条件（乘大数法）,乘大数法施加约束条件，最左侧节点全部固定
	vector<int> cons_DOF;
	for (int i = 0; i < (ele_V + 1) * 2; ++i) {
		cons_DOF.push_back(i);
	}
	VectorXd Disp = VectorXd::Zero(cons_DOF.size());
	for (size_t i = 0; i < cons_DOF.size(); ++i) {
		KZ(cons_DOF[i], cons_DOF[i]) = bt * KZ(cons_DOF[i], cons_DOF[i]);
		PZ(cons_DOF[i]) = Disp(i) * KZ(cons_DOF[i], cons_DOF[i]);
	}

	// 求解位移向量 a,病态
	//VectorXd a = KZ.ldlt().solve(PZ);
	VectorXd a = KZ.lu().solve(PZ);

	//cout << a << endl;

	//// 定义稀疏矩阵
	//SparseMatrix<double> KZ_sparse(total_DOF, total_DOF);
	//// ... 在组装过程中填充 KZ_sparse ...
	//KZ_sparse = KZ.sparseView();

	//// 使用稀疏求解器
	//SparseLU<SparseMatrix<double>> solver;
	//solver.compute(KZ_sparse);
	//if (solver.info() != Success) {
	//	// 分解失败
	//	cout << "Decomposition failed" << endl;
	//	return -1;
	//}
	//VectorXd a = solver.solve(PZ);
	//if (solver.info() != Success) {
	//	// 求解失败
	//	cout << "Solving failed" << endl;
	//	return -1;
	//}
	//cout << a << endl;

	// 计算单元应力
	vector<Vector3d> sgm(ele_num);
	for (int i = 0; i < ele_num; ++i) {
		sgm[i] = stress_calculate(element, node, i, A, D, t, a);
	}

	for (int i = 0; i < ele_num; i++) {
		cout << sgm[i] << endl;
	}

	// 位移放大系数和变形后的坐标
	double SF = 0.2 / a.cwiseAbs().maxCoeff(); // 位移放大系数，可根据需要调整
	vector<Vector2d> new_node(node_num);
	for (int i = 0; i < node_num; ++i) {
		new_node[i] = node[i] + SF * a.segment<2>(2 * i);
	}

	// 将节点坐标、变形后的节点坐标、单元连接关系、应力数据导出到文件
	ofstream node_file("node.csv");
	ofstream new_node_file("new_node.csv");
	ofstream element_file("element.csv");
	ofstream sgm_file("sgm.csv");

	// 导出节点坐标
	for (const auto& n : node) {
		node_file << n(0) << "," << n(1) << endl;
	}
	node_file.close();

	// 导出变形后的节点坐标
	for (const auto& n : new_node) {
		new_node_file << n(0) << "," << n(1) << endl;
	}
	new_node_file.close();

	// 导出单元连接关系（节点索引从 1 开始，以兼容 MATLAB）
	for (const auto& elem : element) {
		element_file << elem[0]+1 << "," << elem[1] + 1 << "," << elem[2] + 1 << endl;
	}
	element_file.close();

	// 导出单元应力（这里以 σ11 为例，即 sgm[0] 分量）
	for (const auto& stress : sgm) {
		sgm_file << stress(0) << endl; // σ11
	}
	sgm_file.close();

	cout << "数据已导出到文件，可以使用 MATLAB 或 Python 进行绘图。" << endl;

	return 0;
}
