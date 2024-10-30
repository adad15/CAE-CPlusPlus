#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <fstream>
#include <iostream>
using namespace std;

struct MyUnit_parameter
{
	int ele_H;		// ˮƽ��Ԫ��
	int ele_V;		// ����Ԫ��
	int ele_num;    // ��Ԫ��
	int node_num;   // �ڵ���,156
	int DOF;		// ÿ���ڵ����ɶ�
};

class Triangular_element
{
public:
	Triangular_element() :E(210e9), u(0.4), q(100), t(0.1), length(5.0), height(1.0) { 
		bt = (3 + 0.5407762) * 1e11; 
		ele_size(0) = 0.2;
		ele_size(1) = 0.2;
	}
	// ��ʼ����Ԫ����
	MyUnit_parameter Init_parameter();
	// ����Ԫ���������
	vector<vector<int>> Assign_node_number(const MyUnit_parameter& unit_par);
	// �����������
	vector<Eigen::Vector2d> Assign_node_coordinates(const MyUnit_parameter& unit_par);

	double Triangle_area() {
		double A = 0.5 * ele_size(0) * ele_size(1);
		return A;
	}
	// ��װȫ�ָնȾ���
	Eigen::MatrixXd global_stiffness_matrix(const MyUnit_parameter& unit_par, const vector<vector<int>>& element,
		const vector<Eigen::Vector2d>& node, double A);
	
	bool Save_global_stiffness_matrix(Eigen::MatrixXd& KZ);
	// ����������ؾ���
	Eigen::VectorXd Set_matrix_load(const MyUnit_parameter& unit_par);
	// ʩ��Լ������
	void Impose_constraints(const MyUnit_parameter& unit_par, Eigen::MatrixXd& KZ, Eigen::VectorXd& PZ);
	// �����λ�ƾ���
	Eigen::VectorXd Solving_displacement(const Eigen::MatrixXd& KZ, const Eigen::VectorXd& PZ) {
		Eigen::VectorXd a = KZ.lu().solve(PZ);
		return a;
	}
	// ���㵥ԪӦ��
	vector<Eigen::Vector3d> Solving_stress(const MyUnit_parameter& unit_par, const Eigen::MatrixXd& KZ, const Eigen::VectorXd& PZ,
		const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node,double A, const Eigen::VectorXd& a) {
		int ele_num = unit_par.ele_num;
		// ���㵥ԪӦ��
		vector<Eigen::Vector3d> sgm(ele_num);
		for (int i = 0; i < ele_num; ++i) {
			sgm[i] = stress_calculate(element, node, i, A, D, t, a);
		}

		for (int i = 0; i < ele_num; i++) {
			cout << sgm[i] << endl;
		}
		return sgm;
	}
	// ����������
	void Derived_result(const MyUnit_parameter& unit_par, const Eigen::VectorXd& a, const vector<vector<int>>& element,
		const vector<Eigen::Vector2d>& node, vector<Eigen::Vector3d>& sgm);

private:
	// ���㵥Ԫ�նȾ���ĺ���
	// element��Ԫ�����Ӿ��󣬴洢ÿ����Ԫ��Ӧ�Ľڵ�����
	// node���ڵ�����������ÿ���ڵ���һ����ά����Vector2d
	// ele_id��������ĵ�Ԫ��ţ������ 0 ��ʼ������
	Eigen::MatrixXd ele_stiff_matrix(const vector<vector<int>>& element, 
		const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t);
	// ���㵥ԪӦ���ĺ���
	Eigen::Vector3d stress_calculate(const vector<vector<int>>& element, 
		const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t, const Eigen::VectorXd& a);
private:
	double E;    // ����ģ������λ Pa
	double u;    // ���ɱȣ�������
	double q;    // ���غɣ�����������λ N/�O
	double t;    // ��ȣ���λ m
	double bt;   // �˴������Ĵ��������������ѱ߽�������Ӧ�ĸնȾ����е�����ɾ������������������Ż��ң���ԭ����ŶԲ��ϣ�����ʹ�ó˴�����
	double length; // ���ȣ���λ m
	double height; // �߶ȣ���λ m
	Eigen::Vector2d ele_size; // ��Ԫ������ֱ�Ǳ߳ߴ�
	Eigen::Matrix3d D;    // Ӧ��Ӧ����� D
};