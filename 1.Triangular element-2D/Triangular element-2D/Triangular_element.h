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
	int ele_H;		// 水平单元数
	int ele_V;		// 竖向单元数
	int ele_num;    // 单元数
	int node_num;   // 节点数,156
	int DOF;		// 每个节点自由度
};

class Triangular_element
{
public:
	Triangular_element() :E(210e9), u(0.4), q(100), t(0.1), length(5.0), height(1.0) { 
		bt = (3 + 0.5407762) * 1e11; 
		ele_size(0) = 0.2;
		ele_size(1) = 0.2;
	}
	// 初始化单元参数
	MyUnit_parameter Init_parameter();
	// 给单元三个结点编号
	vector<vector<int>> Assign_node_number(const MyUnit_parameter& unit_par);
	// 定义结点的坐标
	vector<Eigen::Vector2d> Assign_node_coordinates(const MyUnit_parameter& unit_par);

	double Triangle_area() {
		double A = 0.5 * ele_size(0) * ele_size(1);
		return A;
	}
	// 组装全局刚度矩阵
	Eigen::MatrixXd global_stiffness_matrix(const MyUnit_parameter& unit_par, const vector<vector<int>>& element,
		const vector<Eigen::Vector2d>& node, double A);
	
	bool Save_global_stiffness_matrix(Eigen::MatrixXd& KZ);
	// 定义结点外荷载矩阵
	Eigen::VectorXd Set_matrix_load(const MyUnit_parameter& unit_par);
	// 施加约束条件
	void Impose_constraints(const MyUnit_parameter& unit_par, Eigen::MatrixXd& KZ, Eigen::VectorXd& PZ);
	// 求解结点位移矩阵
	Eigen::VectorXd Solving_displacement(const Eigen::MatrixXd& KZ, const Eigen::VectorXd& PZ) {
		Eigen::VectorXd a = KZ.lu().solve(PZ);
		return a;
	}
	// 计算单元应力
	vector<Eigen::Vector3d> Solving_stress(const MyUnit_parameter& unit_par, const Eigen::MatrixXd& KZ, const Eigen::VectorXd& PZ,
		const vector<vector<int>>& element, const vector<Eigen::Vector2d>& node,double A, const Eigen::VectorXd& a) {
		int ele_num = unit_par.ele_num;
		// 计算单元应力
		vector<Eigen::Vector3d> sgm(ele_num);
		for (int i = 0; i < ele_num; ++i) {
			sgm[i] = stress_calculate(element, node, i, A, D, t, a);
		}

		for (int i = 0; i < ele_num; i++) {
			cout << sgm[i] << endl;
		}
		return sgm;
	}
	// 导出计算结果
	void Derived_result(const MyUnit_parameter& unit_par, const Eigen::VectorXd& a, const vector<vector<int>>& element,
		const vector<Eigen::Vector2d>& node, vector<Eigen::Vector3d>& sgm);

private:
	// 计算单元刚度矩阵的函数
	// element：元素连接矩阵，存储每个单元对应的节点索引
	// node：节点坐标向量，每个节点是一个二维向量Vector2d
	// ele_id：待计算的单元编号（假设从 0 开始计数）
	Eigen::MatrixXd ele_stiff_matrix(const vector<vector<int>>& element, 
		const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t);
	// 计算单元应力的函数
	Eigen::Vector3d stress_calculate(const vector<vector<int>>& element, 
		const vector<Eigen::Vector2d>& node, int ele_id, double A, const Eigen::Matrix3d& D, double t, const Eigen::VectorXd& a);
private:
	double E;    // 弹性模量，单位 Pa
	double u;    // 泊松比，无量纲
	double q;    // 外载荷（面力），单位 N/㎡
	double t;    // 厚度，单位 m
	double bt;   // 乘大数法的大数，正常做法把边界条件对应的刚度矩阵中的行列删除，但是这样做结点标号会乱，和原本标号对不上，所以使用乘大数法
	double length; // 长度，单位 m
	double height; // 高度，单位 m
	Eigen::Vector2d ele_size; // 单元的两个直角边尺寸
	Eigen::Matrix3d D;    // 应力应变矩阵 D
};