#include "Triangular_element.h"

MyUnit_parameter Triangular_element::Init_parameter()
{
	// ��ʼ�� D ����
	D << E / (1 - u * u), u* E / (1 - u * u), 0,
		u* E / (1 - u * u), E / (1 - u * u), 0,
		0, 0, E / (2 * (1 + u));


	MyUnit_parameter unit_par;

	unit_par.ele_H = static_cast<int>(length / ele_size(0)); // ˮƽ��Ԫ��
	unit_par.ele_V = static_cast<int>(height / ele_size(1)); // ����Ԫ��
	unit_par.ele_num = unit_par.ele_H * unit_par.ele_V * 2; // ��Ԫ��
	unit_par.node_num = (unit_par.ele_H + 1) * (unit_par.ele_V + 1); // �ڵ���,156
	unit_par.DOF = 2; // ÿ���ڵ����ɶ�
	return unit_par;
}

vector<vector<int>> Triangular_element::Assign_node_number(const MyUnit_parameter& unit_par)
{
	// ����ÿ����Ԫ�Ľڵ��
	int ele_num = unit_par.ele_num;
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;

	vector<vector<int>> element(ele_num, vector<int>(3));
	int num = 0;
	for (int i = 0; i < ele_H; ++i) {
		for (int j = 0; j < ele_V; ++j) {
			// �����ǣ���ʱ����
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
	// ����ڵ����꣬ʹ�� vector<Vector2d>
	vector<Eigen::Vector2d> node;
	int ele_H = unit_par.ele_H;
	int ele_V = unit_par.ele_V;
	for (int i = 0; i <= ele_H; ++i) {
		for (int j = 0; j <= ele_V; ++j) {
			Eigen::Vector2d coord;
			coord(0) = i * ele_size(0); // x ����
			coord(1) = j * ele_size(1); // y ����
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
	// ÿ����Ԫ���ɶ�����������е�����λ��
	vector<vector<int>> ele_DOF(ele_num, vector<int>(DOF * 3)/*����ά��*/);
	for (int i = 0; i < ele_num; ++i) {
		for (int j = 0; j < 3; ++j) { // �����������ڵ�ѭ��,��ʱ��
			int node_id = element[i][j];
			ele_DOF[i][2 * j] = 2 * node_id;
			ele_DOF[i][2 * j + 1] = 2 * node_id + 1;
		}
	}

	// ��װȫ�ָնȾ���
	int total_DOF = node_num * DOF;
	Eigen::MatrixXd KZ = Eigen::MatrixXd::Zero(total_DOF, total_DOF); //ά��156*2=312

	for (int i = 0; i < ele_num; ++i) {
		Eigen::MatrixXd ke = ele_stiff_matrix(element, node, i, A, D, t); // ��i����Ԫ�ĸնȾ���
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
					KZ_file << ","; // ���ŷָ�
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
	// �����غ����� PZ��ά��312
	Eigen::VectorXd PZ = Eigen::VectorXd::Zero(total_DOF);

	// ʩ�����غɣ���Ҫ���ݾ��������������
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
	// ʩ��Լ���������˴�������,�˴�����ʩ��Լ�������������ڵ�ȫ���̶�
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
	// λ�ƷŴ�ϵ���ͱ��κ������
	double SF = 0.2 / a.cwiseAbs().maxCoeff(); // λ�ƷŴ�ϵ�����ɸ�����Ҫ����
	vector<Eigen::Vector2d> new_node(node_num);
	for (int i = 0; i < node_num; ++i) {
		new_node[i] = node[i] + SF * a.segment<2>(2 * i);
	}

	// ���ڵ����ꡢ���κ�Ľڵ����ꡢ��Ԫ���ӹ�ϵ��Ӧ�����ݵ������ļ�
	ofstream node_file("node.csv");
	ofstream new_node_file("new_node.csv");
	ofstream element_file("element.csv");
	ofstream sgm_file("sgm.csv");

	// �����ڵ�����
	for (const auto& n : node) {
		node_file << n(0) << "," << n(1) << endl;
	}
	node_file.close();

	// �������κ�Ľڵ�����
	for (const auto& n : new_node) {
		new_node_file << n(0) << "," << n(1) << endl;
	}
	new_node_file.close();

	// ������Ԫ���ӹ�ϵ���ڵ������� 1 ��ʼ���Լ��� MATLAB��
	for (const auto& elem : element) {
		element_file << elem[0] + 1 << "," << elem[1] + 1 << "," << elem[2] + 1 << endl;
	}
	element_file.close();

	// ������ԪӦ���������� ��11 Ϊ������ sgm[0] ������
	for (const auto& stress : sgm) {
		sgm_file << stress(0) << endl; // ��11
	}
	sgm_file.close();

	cout << "�����ѵ������ļ�������ʹ�� MATLAB �� Python ���л�ͼ��" << endl;
}

Eigen::MatrixXd Triangular_element::ele_stiff_matrix(const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t)
{
	int i = ele_id; // ���� ele_id �� 0 ��ʼ����

	// ��ȡ��i����Ԫ�Ľڵ�����
	int n1 = element[i][0]; // �ڵ� 1
	int n2 = element[i][1]; // �ڵ� 2
	int n3 = element[i][2]; // �ڵ� 3

	// ��ȡ�ڵ�����
	double x1 = node[n1](0);
	double y1 = node[n1](1);
	double x2 = node[n2](0);
	double y2 = node[n2](1);
	double x3 = node[n3](0);
	double y3 = node[n3](1);

	// ����ϵ��
	double a1 = x2 * y3 - x3 * y2;
	double a2 = x3 * y1 - x1 * y3;
	double a3 = x1 * y2 - x2 * y1;

	double b1 = y2 - y3;
	double b2 = y3 - y1;
	double b3 = y1 - y2;

	double c1 = -x2 + x3;
	double c2 = -x3 + x1;
	double c3 = -x1 + x2;

	// ����Ӧ��-λ�ƾ��� B
	double factor = 1.0 / (2.0 * A);

	Eigen::Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor; //B ��Ϊ������

	// ���㵥Ԫ�նȾ��� ke
	Eigen::MatrixXd ke = B.transpose() * D * B * t * A;

	return ke; // 6*6�ľ���3*2��3����㣬ÿ�����2�����ɶȣ�
}

Eigen::Vector3d Triangular_element::stress_calculate(const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t, const Eigen::VectorXd& a)
{
	int i = ele_id;

	// ��ȡ�� i ����Ԫ�Ľڵ�����
	int n1 = element[i][0];
	int n2 = element[i][1];
	int n3 = element[i][2];

	// ��ȡ�ڵ�����
	double x1 = node[n1](0);
	double y1 = node[n1](1);
	double x2 = node[n2](0);
	double y2 = node[n2](1);
	double x3 = node[n3](0);
	double y3 = node[n3](1);

	// ����ϵ��
	double a1 = x2 * y3 - x3 * y2;
	double a2 = x3 * y1 - x1 * y3;
	double a3 = x1 * y2 - x2 * y1;

	double b1 = y2 - y3;
	double b2 = y3 - y1;
	double b3 = y1 - y2;

	double c1 = -x2 + x3;
	double c2 = -x3 + x1;
	double c3 = -x1 + x2;

	// ����Ӧ��-λ�ƾ��� B
	double factor = 1.0 / (2.0 * A);

	Eigen::Matrix<double, 3, 6> B;
	B << b1, 0, b2, 0, b3, 0,
		0, c1, 0, c2, 0, c3,
		c1, b1, c2, b2, c3, b3;

	B *= factor;

	// ��ȡ�ڵ�λ�� ae
	Eigen::VectorXd ae(6);
	ae << a(2 * n1), a(2 * n1 + 1),
		a(2 * n2), a(2 * n2 + 1),
		a(2 * n3), a(2 * n3 + 1);

	// ���㵥ԪӦ��
	Eigen::Vector3d yps = B * ae;

	// ���㵥ԪӦ��
	Eigen::Vector3d sgm = D * yps;

	return sgm;
}