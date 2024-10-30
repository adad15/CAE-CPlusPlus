#include "Triangular_element.h"

int main() {
	Triangular_element* test=new Triangular_element;

	MyUnit_parameter unit_par = test->Init_parameter();
	vector<vector<int>> element = test->Assign_node_number(unit_par);
	for (int i = 0; i < 250; i++) {
		cout << element[i][0] << " " << element[i][1] << " " << element[i][2] << endl;;
	}

	vector<Eigen::Vector2d> node = test->Assign_node_coordinates(unit_par);
	for (int i = 0; i < node.size(); i++) {
		cout << node[i](0) << " " << node[i](1) << endl;
	}

	double A = test->Triangle_area();
	Eigen::MatrixXd KZ = test->global_stiffness_matrix(unit_par, element, node, A);
	if (test->Save_global_stiffness_matrix(KZ)) {
		cout << "全局刚度矩阵 KZ 已保存到文件 KZ_matrix.csv" << endl;
	}
	else {
		cout << "无法打开文件 KZ_matrix.csv 进行写入。" << endl;
	}
	Eigen::VectorXd PZ = test->Set_matrix_load(unit_par);
	test->Impose_constraints(unit_par, KZ, PZ);
	Eigen::VectorXd a = test->Solving_displacement(KZ, PZ);
	vector<Eigen::Vector3d> sgm = test->Solving_stress(unit_par, KZ, PZ, element, node, A, a);
	test->Derived_result(unit_par, a, element, node, sgm);

	delete test;
}