#include "Triangular_element.h"

MyUnit_parameter Triangular_element::Init_parameter()
{
	// 初始化 D 矩阵
	D << E / (1 - u * u), u* E / (1 - u * u), 0,
		u* E / (1 - u * u), E / (1 - u * u), 0,
		0, 0, E / (2 * (1 + u));


	MyUnit_parameter unit_par;

	unit_par.ele_H = static_cast<int>(length / ele_size(0)); // 水平单元数
	unit_par.ele_V = static_cast<int>(height / ele_size(1)); // 竖向单元数
	unit_par.ele_num = unit_par.ele_H * unit_par.ele_V * 2; // 单元数
	unit_par.node_num = (unit_par.ele_H + 1) * (unit_par.ele_V + 1); // 节点数,156
	unit_par.DOF = 2; // 每个节点自由度
	return unit_par;
}

vector<vector<int>> Triangular_element::Assign_node_number(const MyUnit_parameter& unit_par)
{
	// 定义每个单元的节点号
	int ele_num = unit_par.ele_num;
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;

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
	return element;
}

vector<Eigen::Vector2d> Triangular_element::Assign_node_coordinates(const MyUnit_parameter& unit_par)
{
	// 定义节点坐标，使用 vector<Vector2d>
	vector<Eigen::Vector2d> node;
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;
	for (int i = 0; i <= ele_H; ++i) {
		for (int j = 0; j <= ele_V; ++j) {
			Eigen::Vector2d coord;
			coord(0) = i * ele_size(0); // x 坐标
			coord(1) = j * ele_size(1); // y 坐标
			node.push_back(coord);
		}
	}
	return node;
}

Eigen::MatrixXd Triangular_element::global_stiffness_matrix(const MyUnit_parameter& unit_par, const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, double A)
{
	int ele_num = unit_par.ele_num;
	int node_num = unit_par.node_num;
	int DOF = unit_par.DOF;
	// 每个单元自由度在整体矩阵中的索引位置
	vector<vector<int>> ele_DOF(ele_num, vector<int>(DOF * 3)/*单刚维度*/);
	for (int i = 0; i < ele_num; ++i) {
		for (int j = 0; j < 3; ++j) { // 三角形三个节点循环,逆时针
			int node_id = element[i][j];
			ele_DOF[i][2 * j] = 2 * node_id;
			ele_DOF[i][2 * j + 1] = 2 * node_id + 1;
		}
	}

	// 组装全局刚度矩阵
	int total_DOF = node_num * DOF;
	Eigen::MatrixXd KZ = Eigen::MatrixXd::Zero(total_DOF, total_DOF); //维度156*2=312

	for (int i = 0; i < ele_num; ++i) {
		Eigen::MatrixXd ke = ele_stiff_matrix(element, node, i, A, D, t); // 第i个单元的刚度矩阵
		for (int m = 0; m < ke.rows(); ++m) {
			int row = ele_DOF[i][m];
			for (int n = 0; n < ke.cols(); ++n) {
				int col = ele_DOF[i][n];
				KZ(row, col) += ke(m, n);
			}
		}
	}
	return KZ;
}

bool Triangular_element::Save_global_stiffness_matrix(Eigen::MatrixXd& KZ)
{
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
		return true;
	}
	else {
		return false;
	}
}

Eigen::VectorXd Triangular_element::Set_matrix_load(const MyUnit_parameter& unit_par)
{
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;
	int node_num = unit_par.node_num;
	int DOF = unit_par.DOF;
	int total_DOF = node_num * DOF;
	// 定义载荷向量 PZ，维度312
	Eigen::VectorXd PZ = Eigen::VectorXd::Zero(total_DOF);

	// 施加外载荷（需要根据具体的索引调整）
	PZ(2 * ele_V + 1) = -0.5 * (q * t * length) / ele_H;
	for (int i = 1; i < ele_H; ++i) {
		int idx = 11 + 2 * (ele_V + 1) * i;
		if (idx < total_DOF) {
			PZ(idx) = -q * t * length / ele_H;
		}
	}
	PZ(total_DOF - 1) = -0.5 * q * t * 5 / 10;

	return PZ;
}

void Triangular_element::Impose_constraints(const MyUnit_parameter& unit_par, Eigen::MatrixXd& KZ, Eigen::VectorXd& PZ)
{
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;
	// 施加约束条件（乘大数法）,乘大数法施加约束条件，最左侧节点全部固定
	vector<int> cons_DOF;
	for (int i = 0; i < (ele_V + 1) * 2; ++i) {
		cons_DOF.push_back(i);
	}
	Eigen::VectorXd Disp = Eigen::VectorXd::Zero(cons_DOF.size());
	for (size_t i = 0; i < cons_DOF.size(); ++i) {
		KZ(cons_DOF[i], cons_DOF[i]) = bt * KZ(cons_DOF[i], cons_DOF[i]);
		PZ(cons_DOF[i]) = Disp(i) * KZ(cons_DOF[i], cons_DOF[i]);
	}
}

void Triangular_element::Derived_result(const MyUnit_parameter& unit_par, const Eigen::VectorXd& a, const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, vector<Eigen::Vector3d>& sgm)
{
	int node_num = unit_par.node_num;
	// 位移放大系数和变形后的坐标
	double SF = 0.2 / a.cwiseAbs().maxCoeff(); // 位移放大系数，可根据需要调整
	vector<Eigen::Vector2d> new_node(node_num);
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
		element_file << elem[0] + 1 << "," << elem[1] + 1 << "," << elem[2] + 1 << endl;
	}
	element_file.close();

	// 导出单元应力（这里以 σ11 为例，即 sgm[0] 分量）
	for (const auto& stress : sgm) {
		sgm_file << stress(0) << endl; // σ11
	}
	sgm_file.close();

	cout << "数据已导出到文件，可以使用 MATLAB 或 Python 进行绘图。" << endl;
}

Eigen::MatrixXd Triangular_element::ele_stiff_matrix(const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t)
{
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

	Eigen::Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor; //B 阵为常数阵

	// 计算单元刚度矩阵 ke
	Eigen::MatrixXd ke = B.transpose() * D * B * t * A;

	return ke; // 6*6的矩阵，3*2（3个结点，每个结点2个自由度）
}

Eigen::Vector3d Triangular_element::stress_calculate(const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t, const Eigen::VectorXd& a)
{
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

	Eigen::Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor;

	// 提取节点位移 ae
	Eigen::VectorXd ae(6);
	ae << a(2 * n1), a(2 * n1 + 1),
		a(2 * n2), a(2 * n2 + 1),
		a(2 * n3), a(2 * n3 + 1);

	// 计算单元应变
	Eigen::Vector3d yps = B * ae;

	// 计算单元应力
	Eigen::Vector3d sgm = D * yps;

	return sgm;
}