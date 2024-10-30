#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
using namespace std;

// 定义稀疏矩阵类型
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::Triplet<double> Triplet;

struct MeshData {
	vector<vector<double>> Nodes;
	vector<vector<int>> Elements;
};

class My_C3D8
{
public:
	void Init() {
		// 初始化 D 矩阵
		double E = 210000;
		double u = 0.3;
		double G = E / (2.0 + 2.0 * u);
		double a = 4 * G / 3;
		double b = -2 * G / 3;
		double c = G;
		D_dev << a, b, b, 0, 0, 0,
				 b, a, b, 0, 0, 0,
				 b, b, a, 0, 0, 0,
				 0, 0, 0, c, 0, 0,
				 0, 0, 0, 0, c, 0,
				 0, 0, 0, 0, 0, c;
		double k = E / (3 * (1 - 2 * u));
		D_dil << k, k, k, 0, 0, 0,
				 k, k, k, 0, 0, 0,
				 k, k, k, 0, 0, 0,
				 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0;
		double denominator = (1.0 + u) * (1.0 - 2.0 * u);
		double coef = E / denominator;
		double a1 = coef * (1.0 - u);
		double b1 = coef * u;
		double c1 = E / (2.0 * (1.0 + u));
		D << a1, b1, b1, 0, 0, 0,
		 	 b1, a1, b1, 0, 0, 0,
			 b1, b1, a1, 0, 0, 0,
			 0, 0, 0, c1, 0, 0,
			 0, 0, 0, 0, c1, 0,
			 0, 0, 0, 0, 0, c1;
	}
	// 该函数可读取多种类型单元(三角形、四边形、三棱柱、六面体)的节点及单元信息
	MeshData Readmesh(const string& fname) {
		ifstream fid(fname);
		if (!fid.is_open()) {
			cerr << "无法打开文件 " << fname << endl;
			exit(1);
		}
		// 将所有行读取到字符串向量 S 中
		vector<string> S;
		string line;
		while (getline(fid, line)) {
			S.push_back(line);
		}
		fid.close();

		// 查找包含关键字 "Node"、"Element" 和 "Nset" 的行索引
		size_t idx_Node = string::npos;
		size_t idx_Element = string::npos;
		size_t idx_Nset = string::npos;

		for (size_t i = 0; i < S.size(); ++i) {
			if (S[i].find("Node") != string::npos && idx_Node == string::npos) {
				idx_Node = i;
			}
			if (S[i].find("Element") != string::npos && idx_Element == string::npos) {
				idx_Element = i;
			}
			if (S[i].find("Nset") != string::npos && idx_Nset == string::npos) {
				idx_Nset = i;
			}
		}

		// 检查是否找到所有关键字
		if (idx_Node == string::npos || idx_Element == string::npos || idx_Nset == string::npos) {
			cerr << "文件中未找到必要的部分。" << endl;
			exit(1);
		}

		// 提取节点和单元信息
		vector<string> Nodes_lines(S.begin() + idx_Node + 1, S.begin() + idx_Element);
		vector<string> Elements_lines(S.begin() + idx_Element + 1, S.begin() + idx_Nset);

		// 解析节点信息
		vector<vector<double>> Nodes;
		for (const auto& str : Nodes_lines) {
			vector<double> numbers = parseNumbers(str);
			if (!numbers.empty() && numbers.size() > 1) {
				// 移除第一个数字（节点编号）
				numbers.erase(numbers.begin());
				Nodes.push_back(numbers);
			}
		}

		// 解析单元信息
		vector<vector<int>> Elements;
		for (const auto& str : Elements_lines) {
			vector<int> numbers = parseIntegers(str);
			if (!numbers.empty() && numbers.size() > 1) {
				// 移除第一个数字（单元编号）
				numbers.erase(numbers.begin());
				Elements.push_back(numbers);
			}
		}

		// 返回节点和单元数据
		MeshData mesh;
		mesh.Nodes = Nodes;
		mesh.Elements = Elements;
		return mesh;
	}
	/*
	 *  N 局部坐标下的形函数矩阵
	 *  NDerivative形函数矩阵对全局坐标的导数
	 * 
	 *  JacobiDET雅可比行列式
	 *  GaussPoint高斯点坐标
	 *  ElementNodeCoordinate单元节点坐标（8*3，每一行代表一个节点的坐标）
	 * 
	 */
	void ShapeFunction(const vector<double>& GaussPoint, const vector<vector<double>>& ElementNodeCoordinate/* 全局坐标 */,
		vector<double>& N, vector<vector<double>>& NDerivative, double& JacobiDET) {
		// 定义等参元节点坐标（3x8 矩阵）每一列代表一个点的坐标，自然坐标系 维度：3*8
		vector<vector<double>> ParentNodes = {
			{-1, 1, 1, -1, -1, 1, 1, -1},
			{-1, -1, 1, 1, -1, -1, 1, 1},
			{-1, -1, -1, -1, 1, 1, 1, 1}
		};

		N.resize(8, 0.0); // 初始化形函数矩阵 8x1
		vector<vector<double>> ParentNDerivative(3, vector<double>(8, 0.0)); // 形函数对局部坐标的导数矩阵 维度：3x8

		// 计算形函数及其对局部坐标的导数
		for (int I = 0; I < 8; ++I) {
			double XPoint = ParentNodes[0][I];
			double YPoint = ParentNodes[1][I];
			double ZPoint = ParentNodes[2][I];
			vector<double> ShapePart = {
				1 + GaussPoint[0] * XPoint,
				1 + GaussPoint[1] * YPoint,
				1 + GaussPoint[2] * ZPoint
			};
			// 形函数在高斯点上的值
			N[I] = 0.125 * ShapePart[0] * ShapePart[1] * ShapePart[2];
			// 形函数对等参坐标的导数
			ParentNDerivative[0][I] = 0.125 * XPoint * ShapePart[1] * ShapePart[2];
			ParentNDerivative[1][I] = 0.125 * YPoint * ShapePart[0] * ShapePart[2];
			ParentNDerivative[2][I] = 0.125 * ZPoint * ShapePart[0] * ShapePart[1];
		}

		// 计算雅可比矩阵（3x3 矩阵）
		vector<vector<double>> Jacobi(3, vector<double>(3, 0.0));
		for (int i = 0; i < 3; ++i) { // 对于行
			for (int j = 0; j < 3; ++j) { // 对于列
				for (int k = 0; k < 8; ++k) {
					Jacobi[i][j] += ParentNDerivative[i][k] * ElementNodeCoordinate[k][j];
				}
			}
		}

		// 计算雅可比行列式
		JacobiDET = determinant3x3(Jacobi);

		// 计算雅可比矩阵的逆
		vector<vector<double>> JacobiINV = inverse3x3(Jacobi);

		// 计算形函数对全局坐标的导数（3x8 矩阵）
		NDerivative.resize(3, std::vector<double>(8, 0.0));
		for (int i = 0; i < 3; ++i) { // 行
			for (int j = 0; j < 8; ++j) { // 列
				for (int k = 0; k < 3; ++k) {
					NDerivative[i][j] += JacobiINV[i][k] * ParentNDerivative[k][j];
				}
			}
		}
	}

	// 计算单元刚度矩阵 Ke
	void Ke(const vector<vector<double>>& ElementNodeCoordinate,vector<vector<double>>& Ke) {
		// 高斯积分点坐标
		vector<double> GaussCoordinate = { -0.57735026918963, 0.57735026918963 };
		// 高斯积分点权重
		vector<double> GaussWeight = { 1.0, 1.0 };

		// 初始化 Ke 为 24x24 的零矩阵
		Ke.assign(24, vector<double>(24, 0.0));

		// 偏刚度完全积分
		for (int X = 0; X < 2; ++X) {
			for (int Y = 0; Y < 2; ++Y) {
				for (int Z = 0; Z < 2; ++Z) {
					double GP1 = GaussCoordinate[X];
					double GP2 = GaussCoordinate[Y];
					double GP3 = GaussCoordinate[Z];

					vector<double> GaussPoint = { GP1, GP2, GP3 };

					// 声明 N, NDerivative, JacobiDET
					vector<double> N;
					// 形函数对全局坐标的导数，维度：3x8
					vector<vector<double>> NDerivative;
					double JacobiDET;

					// 调用 ShapeFunction 函数
					ShapeFunction(GaussPoint, ElementNodeCoordinate, N, NDerivative, JacobiDET);

					double Coefficient = GaussWeight[X] * GaussWeight[Y] * GaussWeight[Z] * JacobiDET;

					// 初始化 B 为 6x24 的零矩阵
					vector<vector<double>> B(6, vector<double>(24, 0.0));

					// 构建 B 矩阵，以列为循环计算
					for (int I = 0; I < 8; ++I) {
						int COL = I * 3; // 3代表xyz

						// 填充 B 矩阵对应的列
						B[0][COL] = NDerivative[0][I];
						B[1][COL + 1] = NDerivative[1][I];
						B[2][COL + 2] = NDerivative[2][I];

						B[3][COL] = NDerivative[1][I];
						B[3][COL + 1] = NDerivative[0][I];

						B[4][COL + 1] = NDerivative[2][I];
						B[4][COL + 2] = NDerivative[1][I];

						B[5][COL] = NDerivative[2][I];
						B[5][COL + 2] = NDerivative[0][I];
					}

					// 计算 temp = D（6*6） * B（6*24）
					vector<vector<double>> temp(6, vector<double>(24, 0.0));
					for (int i = 0; i < 6; ++i) {
						for (int j = 0; j < 24; ++j) {
							for (int k = 0; k < 6; ++k) {
								temp[i][j] += D_dev(i, k) * B[k][j];
							}
						}
					}

					// 计算 Ke 增量：Ke_increment = Coefficient * B^T * temp（24*24）
					vector<vector<double>> Ke_increment(24, vector<double>(24, 0.0));
					for (int i = 0; i < 24; ++i) {
						for (int j = 0; j < 24; ++j) {
							for (int k = 0; k < 6; ++k) {
								Ke_increment[i][j] += B[k][i]/* 转置 */ * temp[k][j];
							}
							Ke_increment[i][j] *= Coefficient;
						}
					}

					// 累加到 Ke
					for (int i = 0; i < 24; ++i) {
						for (int j = 0; j < 24; ++j) {
							Ke[i][j] += Ke_increment[i][j];
						}
					}
				}
			}
		}
		// 体积刚度减缩积分部分,降一阶
		double GP1 = 0.0, GP2 = 0.0, GP3 = 0.0; // 使用中心点
		vector<double> GaussPoint = { GP1, GP2, GP3 };
		// 声明 N, NDerivative, JacobiDET
		vector<double> N;
		vector<vector<double>> NDerivative;
		double JacobiDET;

		// 调用 ShapeFunction 函数
		ShapeFunction(GaussPoint, ElementNodeCoordinate, N, NDerivative, JacobiDET);

		// 初始化 B 矩阵
		vector<vector<double>> B(6, vector<double>(24, 0.0));

		// 构建 B 矩阵
		for (int I = 0; I < 8; ++I) {
			int COL = I * 3; // 3代表xyz
			B[0][COL] = NDerivative[0][I];
			B[1][COL + 1] = NDerivative[1][I];
			B[2][COL + 2] = NDerivative[2][I];
			B[3][COL] = NDerivative[1][I];
			B[3][COL + 1] = NDerivative[0][I];
			B[4][COL + 1] = NDerivative[2][I];
			B[4][COL + 2] = NDerivative[1][I];
			B[5][COL] = NDerivative[2][I];
			B[5][COL + 2] = NDerivative[0][I];
		}
		// 体积刚度矩阵计算
		double gaussWeight = 4.0; // MATLAB中的权重
		double Coefficient = gaussWeight * JacobiDET;

		// 计算 temp = D（6*6） * B（6*24）
		vector<vector<double>> temp(6, vector<double>(24, 0.0));
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 24; ++j) {
				for (int k = 0; k < 6; ++k) {
					temp[i][j] += D_dil(i, k) * B[k][j];
				}
			}
		}

		// 计算 Ke 增量：Ke_increment = Coefficient * B^T * temp（24*24）
		vector<vector<double>> Ke_increment(24, vector<double>(24, 0.0));
		for (int i = 0; i < 24; ++i) {
			for (int j = 0; j < 24; ++j) {
				for (int k = 0; k < 6; ++k) {
					Ke_increment[i][j] += B[k][i]/* 转置 */ * temp[k][j];
				}
				Ke_increment[i][j] *= Coefficient;
			}
		}

		// 累积体积刚度 Ke_increment
		for (int i = 0; i < 24; ++i) {
			for (int j = 0; j < 24; ++j) {
				// 将体积刚度积分累加到 Ke
				Ke[i][j] += Ke_increment[i][j];
			}
		}
	}

	// 计算位移矩阵 U
	void StaticsSolver(double E, double u,
		const vector<vector<double>>& Forces,
		const vector<vector<double>>& Constraints,
		const vector<vector<double>>& Nodes,
		const vector<vector<int>>& Elements,
		vector<double>& U) {
		int Dof = 3;
		int NodeCount = Nodes.size();        // 节点个数
		int ElementCount = Elements.size();  // 单元个数
		int Dofs = Dof * NodeCount;          // 总自由度数

		U.resize(Dofs, 0.0);             // 初始化位移向量 U
		vector<Triplet> K_triplets;      // 用于构建稀疏矩阵 K
		Eigen::VectorXd Force = Eigen::VectorXd::Zero(Dofs); // 初始化外力向量

		// 组装总体刚度矩阵 K
		for (int I = 0; I < ElementCount; ++I) {
			// 单元节点坐标（8x3）
			vector<vector<double>> ElementNodeCoordinate(8, vector<double>(3, 0.0));
			for (int j = 0; j < 8; ++j) {
				int nodeIndex = Elements[I][j] - 1; // 节点编号从 0 开始，inp 从 1 开始
				ElementNodeCoordinate[j] = Nodes[nodeIndex];
			}

			// 计算单元刚度矩阵（24x24）
			vector<vector<double>> ElementStiffnessMatrix;
			Ke(ElementNodeCoordinate, ElementStiffnessMatrix);

			// 计算单元节点自由度编号
			vector<int> ElementNodeDOF(24, 0);
			for (int J = 0; J < 8; ++J) {
				int base = J * Dof;
				int nodeIndex = Elements[I][J] - 1;
				ElementNodeDOF[base] = nodeIndex * Dof;
				ElementNodeDOF[base + 1] = nodeIndex * Dof + 1;
				ElementNodeDOF[base + 2] = nodeIndex * Dof + 2;
			}

			// 将单元刚度矩阵组装到总体刚度矩阵 K 中
			for (int i = 0; i < 24; ++i) {
				int row = ElementNodeDOF[i];
				for (int j = 0; j < 24; ++j) {
					int col = ElementNodeDOF[j];
					double value = ElementStiffnessMatrix[i][j];
					if (fabs(value) > 1e-12) {
						K_triplets.emplace_back(row, col, value);
					}
				}
			}
		}

		// 构建稀疏矩阵 K
		SparseMatrix K(Dofs, Dofs);
		K.setFromTriplets(K_triplets.begin(), K_triplets.end());

		// 施加外力
		if (!Forces.empty()) {
			for (const auto& force : Forces) {
				int node = static_cast<int>(force[0]) - 1;
				int direction = static_cast<int>(force[1]) - 1;
				double magnitude = force[2];
				int dofIndex = node * Dof + direction;
				Force(dofIndex) += magnitude;
			}
		}

		// 乘大数法施加位移约束
		double BigNumber = 1e20;
		if (!Constraints.empty()) {
			for (const auto& constraint : Constraints) {
				int node = static_cast<int>(constraint[0]) - 1;
				int direction = static_cast<int>(constraint[1]) - 1;
				double magnitude = constraint[2];
				int dofIndex = node * Dof + direction;

				// 修改刚度矩阵 K 和外力向量 Force
				K.coeffRef(dofIndex, dofIndex) *= BigNumber;
				Force(dofIndex) = magnitude * K.coeffRef(dofIndex, dofIndex);
			}
		}

		// 求解线性方程组 K * U = Force
		Eigen::SimplicialLDLT<SparseMatrix> solver;
		solver.compute(K);
		if (solver.info() != Eigen::Success) {
			cerr << "无法分解刚度矩阵 K。" << endl;
			return;
		}
		Eigen::VectorXd U_eigen = solver.solve(Force);
		if (solver.info() != Eigen::Success) {
			cerr << "无法求解线性方程组。" << endl;
			return;
		}

		// 将结果复制到输出向量 U 中
		for (int i = 0; i < Dofs; ++i) {
			U[i] = U_eigen[i];
		}
	}

	void CalculateStrainAndStress(const vector<double>& U,
		const vector<vector<double>>& Nodes,
		const vector<vector<int>>& Elements,
		vector<Eigen::VectorXd>& NodeStrain,
		vector<Eigen::VectorXd>& NodeStress,
		vector<Eigen::VectorXd>& GaussStrain,
		vector<Eigen::VectorXd>& GaussStress) {
		int ElementCount = Elements.size();    // 单元个数
		vector<double> GaussCoordinate = { -0.57735026918963, 0.57735026918963 };  // 高斯积分点坐标
		vector<double> GaussWeight = { 1.0, 1.0 };  // 高斯积分点权重
		int Dof = 3;
		int GaussPointTotal = ElementCount * 8; // 总的高斯点数量
		int NodeTotal = Nodes.size(); // 节点总数

		// 预分配 GaussStrain 和 GaussStress  6个应力应变
		GaussStrain.resize(GaussPointTotal, Eigen::VectorXd::Zero(6));
		GaussStress.resize(GaussPointTotal, Eigen::VectorXd::Zero(6));
		NodeStrain.resize(NodeTotal, Eigen::VectorXd::Zero(6));
		NodeStress.resize(NodeTotal, Eigen::VectorXd::Zero(6));

		int GaussPointNumber = 0; // 高斯积分点编号，最大值为单元个数*8

		// 循环每个单元
		for (int I = 0; I < ElementCount; ++I) {
			// 单元节点坐标（8x3）
			std::vector<std::vector<double>> ElementNodeCoordinate(8, std::vector<double>(3, 0.0));
			std::vector<int> ElementNodeIndices(8); // 节点全局编号
			for (int j = 0; j < 8; ++j) {
				int nodeIndex = Elements[I][j] - 1; 
				ElementNodeCoordinate[j] = Nodes[nodeIndex];
				ElementNodeIndices[j] = nodeIndex;
			}

			// 计算单元节点自由度编号
			std::vector<int> ElementNodeDOF(24, 0);
			for (int J = 0; J < 8; ++J) {
				int II = J * Dof;
				int nodeIndex = ElementNodeIndices[J];
				ElementNodeDOF[II] = nodeIndex * Dof;
				ElementNodeDOF[II + 1] = nodeIndex * Dof + 1;
				ElementNodeDOF[II + 2] = nodeIndex * Dof + 2;
			}

			int K = 0; // 高斯积分点编号，最大为8
			Eigen::MatrixXd InterpolationMatrix(8/*8个高斯积分点*/, 8/*8个结点*/);
			InterpolationMatrix.setZero();

			// 临时存储该单元的 GaussStrain 和 GaussStress
			Eigen::MatrixXd GaussStrainElement(6, 8);
			Eigen::MatrixXd GaussStressElement(6, 8);

			// 循环高斯点
			for (int X = 0; X < 2; ++X) {
				for (int Y = 0; Y < 2; ++Y) {
					for (int Z = 0; Z < 2; ++Z) {
						double E1 = GaussCoordinate[X];
						double E2 = GaussCoordinate[Y];
						double E3 = GaussCoordinate[Z];

						// 计算形函数 N 和导数 NDerivative -—— 形函数对全局坐标的导数
						std::vector<double> N;
						std::vector<std::vector<double>> NDerivative;
						double JacobiDET;
						ShapeFunction({ E1, E2, E3 }, ElementNodeCoordinate, N, NDerivative, JacobiDET);

						// 获取单元节点位移
						Eigen::VectorXd ElementNodeDisplacement(24);
						for (int i = 0; i < 24; ++i) {
							ElementNodeDisplacement(i) = U[ElementNodeDOF[i]];
						}

						// 将位移重塑为 3x8 的矩阵
						Eigen::Matrix<double, 3, 8> DisplacementMatrix;
						for (int col = 0; col < 8; ++col) {
							DisplacementMatrix(0, col) = ElementNodeDisplacement(col * 3);
							DisplacementMatrix(1, col) = ElementNodeDisplacement(col * 3 + 1);
							DisplacementMatrix(2, col) = ElementNodeDisplacement(col * 3 + 2);
						}

						// 形函数对全局坐标的导数 NDerivative 转换为 Eigen 矩阵
						Eigen::Matrix<double, 3, 8> NDerivativeMatrix;
						for (int i = 0; i < 3; ++i)
							for (int j = 0; j < 8; ++j)
								NDerivativeMatrix(i, j) = NDerivative[i][j];

						// 计算高斯点应变矩阵（3x3）
						Eigen::Matrix<double, 3, 3> GausspointStrain3_3 = DisplacementMatrix * NDerivativeMatrix.transpose()/*行向量显示*/;

						// 将应变矩阵转换为 6x1 的向量
						Eigen::Matrix<double, 6, 1> GausspointStrain;
						GausspointStrain(0) = GausspointStrain3_3(0, 0); // ε_xx
						GausspointStrain(1) = GausspointStrain3_3(1, 1); // ε_yy
						GausspointStrain(2) = GausspointStrain3_3(2, 2); // ε_zz
						GausspointStrain(3) = GausspointStrain3_3(0, 1) + GausspointStrain3_3(1, 0); // γ_xy
						GausspointStrain(4) = GausspointStrain3_3(1, 2) + GausspointStrain3_3(2, 1); // γ_yz
						GausspointStrain(5) = GausspointStrain3_3(0, 2) + GausspointStrain3_3(2, 0); // γ_xz

						// 计算高斯点应力
						Eigen::Matrix<double, 6, 1> GausspointStress = D * GausspointStrain;

						// 存储高斯点的应变和应力
						GaussStrain[GaussPointNumber] = GausspointStrain;
						GaussStress[GaussPointNumber] = GausspointStress;

						// 将 N 存入插值矩阵
						for (int i = 0; i < 8; ++i) {
							InterpolationMatrix(K, i) = N[i];
						}

						K++;
						GaussPointNumber++;
					}
				}
			}

			// 求解节点应变和应力
			// InterpolationMatrix 为 8x8 矩阵，GaussStrainElement^T 为 8x6 矩阵
			// 需要求解 InterpolationMatrix * NodeStrainElement = GaussStrainElement^T

			// 将当前单元的 GaussStrain 和 GaussStress 提取出来
			Eigen::MatrixXd GaussStrainElement_T(8, 6);
			Eigen::MatrixXd GaussStressElement_T(8, 6);
			for (int k = 0; k < 8; ++k) {
				GaussStrainElement_T.row(k) = GaussStrain[GaussPointNumber - 8 + k];
				GaussStressElement_T.row(k) = GaussStress[GaussPointNumber - 8 + k];
			}

			// 求解节点应变和应力（8x6 矩阵）
			Eigen::MatrixXd NodeStrainElement(8, 6);
			Eigen::MatrixXd NodeStressElement(8, 6);

			// 使用线性求解器求解
			Eigen::FullPivLU<Eigen::MatrixXd> solver(InterpolationMatrix);
			NodeStrainElement = solver.solve(GaussStrainElement_T);
			NodeStressElement = solver.solve(GaussStressElement_T);

			// 将节点应变和应力存入全局变量中
			for (int i = 0; i < 8; ++i) {
				int nodeGlobalIndex = ElementNodeIndices[i];
				NodeStrain[nodeGlobalIndex] += NodeStrainElement.row(i).transpose();
				NodeStress[nodeGlobalIndex] += NodeStressElement.row(i).transpose();
			}
		}

		// 平均化节点应变和应力（如果一个节点属于多个单元）
		// 这里假设每个节点的应变和应力已累加，需要除以参与累加的次数
		vector<int> NodeContributionCount(NodeTotal, 0);
		for (int I = 0; I < ElementCount; ++I) {
			for (int j = 0; j < 8; ++j) {
				int nodeIndex = Elements[I][j] - 1;
				NodeContributionCount[nodeIndex]++;
			}
		}
		for (int i = 0; i < NodeTotal; ++i) {
			if (NodeContributionCount[i] > 0) {
				NodeStrain[i] /= NodeContributionCount[i];
				NodeStress[i] /= NodeContributionCount[i];
			}
		}
	}

	void OutputResults(const string& outputFileName,
		const vector<vector<double>>& Nodes,
		const vector<vector<int>>& Elements,
		const vector<double>& U) {
		int NodeCount = Nodes.size();          // 节点个数
		int ElementCount = Elements.size();    // 单元个数
		int ElementNodeCount = 8;              // 每个单元的节点数
		int Dof = 3;

		// 计算高斯点和节点的应变和应力
		vector<Eigen::VectorXd> NodeStrain;
		//（S11, S22, S33, S12, S23, S13）
		vector<Eigen::VectorXd> NodeStress;
		vector<Eigen::VectorXd> GaussStrain;
		vector<Eigen::VectorXd> GaussStress;

		CalculateStrainAndStress(U, Nodes, Elements, NodeStrain, NodeStress, GaussStrain, GaussStress);

		// 计算 Mises 应力矩阵
		vector<double> MISES(NodeCount, 0.0);
		for (int I = 0; I < NodeCount; ++I) {
			double s1 = NodeStress[I](0); // S11
			double s2 = NodeStress[I](1); // S22
			double s3 = NodeStress[I](2); // S33
			double s12 = NodeStress[I](3); // S12
			double s23 = NodeStress[I](4); // S23
			double s13 = NodeStress[I](5); // S13

			MISES[I] = std::sqrt(0.5 * ((s1 - s2) * (s1 - s2) + (s1 - s3) * (s1 - s3) + (s2 - s3) * (s2 - s3)
				+ 6.0 * (s12 * s12 + s23 * s23 + s13 * s13)));
		}

		// 计算位移模量 Umag
		vector<double> Umag(NodeCount, 0.0);
		for (int i = 0; i < NodeCount; ++i) {
			double u1 = U[i * 3];
			double u2 = U[i * 3 + 1];
			double u3 = U[i * 3 + 2];
			Umag[i] = sqrt(u1 * u1 + u2 * u2 + u3 * u3);
		}

		// 将节点位移和应力应变写入 txt 文件
		ofstream outFile(outputFileName);
		if (!outFile.is_open()) {
			cerr << "无法打开输出文件：" << outputFileName << endl;
			return;
		}

		// 写入节点位移
		outFile << "Node          U1          U2          U3\n";
		for (int I = 0; I < NodeCount; ++I) {
			double u1 = U[I * 3];
			double u2 = U[I * 3 + 1];
			double u3 = U[I * 3 + 2];
			outFile << I + 1 << " " << u1 << " " << u2 << " " << u3 << "\n";
		}

		// 写入高斯点应变
		outFile << "\nElement GaussStrain\n";
		outFile << "        E11         E22         E33         E12         E23         E13\n";
		int GaussPointNumber = 0;
		for (int I = 0; I < ElementCount; ++I) {
			outFile << "Element " << I + 1 << "\n";
			for (int j = 0; j < 8; ++j) { // 每个单元有 8 个高斯点
				Eigen::VectorXd strain = GaussStrain[GaussPointNumber++];
				outFile << strain.transpose() << "\n";
			}
		}

		// 写入高斯点应力
		outFile << "\nElement GaussStress\n";
		outFile << "        S11         S22         S33         S12         S23         S13\n";
		GaussPointNumber = 0;
		for (int I = 0; I < ElementCount; ++I) {
			outFile << "Element " << I + 1 << "\n";
			for (int j = 0; j < 8; ++j) { // 每个单元有 8 个高斯点
				Eigen::VectorXd stress = GaussStress[GaussPointNumber++];
				outFile << stress.transpose() << "\n";
			}
		}

		outFile.close();

		// 输出完成信息
		cout << "\t\t *** Successful end of program ***\n";

		// 绘制未变形的网格（将数据输出到文件，供后续绘图）
		ofstream meshFile("mesh_data.txt");
		if (!meshFile.is_open()) {
			cerr << "无法打开网格数据文件。\n";
			return;
		}

		for (const auto& element : Elements) {
			for (int idx : element) {
				const auto& node = Nodes[idx - 1]; // 节点编号从 1 开始
				meshFile << node[0] << " " << node[1] << " " << node[2] << "\n";
			}
			meshFile << "\n"; // 每个单元之间空一行
		}
		meshFile.close();

		// 输出位移、应力、应变等数据到文件，供后续绘制云图
		// 输出 Umag 位移模量
		ofstream umagFile("Umag_data.txt");
		if (!umagFile.is_open()) {
			cerr << "无法打开 Umag 数据文件。\n";
			return;
		}
		for (int i = 0; i < NodeCount; ++i) {
			const auto& node = Nodes[i];
			umagFile << node[0] << " " << node[1] << " " << node[2] << " " << Umag[i] << "\n";
		}
		umagFile.close();

		// 输出 MISES 应力
		ofstream misesFile("Mises_data.txt");
		if (!misesFile.is_open()) {
			cerr << "无法打开 Mises 数据文件。\n";
			return;
		}
		for (int i = 0; i < NodeCount; ++i) {
			const auto& node = Nodes[i];
			misesFile << node[0] << " " << node[1] << " " << node[2] << " " << MISES[i] << "\n";
		}
		misesFile.close();

		// 其他需要输出的数据，如 U1、U2、U3、应力、应变等，可以按照上述方式输出到文件
		// 之后可以使用 Python、MATLAB、ParaView 等工具读取这些数据文件，进行可视化

		// 例如，输出 U1、U2、U3
		ofstream uFile("Displacement_data.txt");
		if (!uFile.is_open()) {
			cerr << "无法打开位移数据文件。\n";
			return;
		}
		for (int i = 0; i < NodeCount; ++i) {
			const auto& node = Nodes[i];
			double u1 = U[i * 3];
			double u2 = U[i * 3 + 1];
			double u3 = U[i * 3 + 2];
			uFile << node[0] << " " << node[1] << " " << node[2] << " " << u1 << " " << u2 << " " << u3 << "\n";
		}
		uFile.close();

		// 输出节点应力（例如 S11）
		ofstream s11File("Stress_S11_data.txt");
		if (!s11File.is_open()) {
			cerr << "无法打开 S11 数据文件。\n";
			return;
		}
		for (int i = 0; i < NodeCount; ++i) {
			const auto& node = Nodes[i];
			double s11 = NodeStress[i](0);
			s11File << node[0] << " " << node[1] << " " << node[2] << " " << s11 << "\n";
		}
		s11File.close();
	}
private:
	vector<double> parseNumbers(const string& str) {
		vector<double> numbers;
		string s = str;
		replace(s.begin(), s.end(), ',', ' '); // 将逗号替换为空格
		istringstream iss(s);
		double num;
		while (iss >> num) {
			numbers.push_back(num);
		}
		return numbers;
	}

	vector<int> parseIntegers(const string& str) {
		vector<int> numbers;
		string s = str;
		replace(s.begin(), s.end(), ',', ' '); // 将逗号替换为空格
		istringstream iss(s);
		int num;
		while (iss >> num) {
			numbers.push_back(num);
		}
		return numbers;
	}

	// 计算 3x3 矩阵的行列式
	double determinant3x3(const vector<vector<double>>& mat) {
		return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
			mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
			mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
	}

	// 计算 3x3 矩阵的逆矩阵,伴随矩阵的方法求解
	vector<vector<double>> inverse3x3(const vector<vector<double>>& mat) {
		double det = determinant3x3(mat);
		if (fabs(det) < 1e-12) {
			cerr << "雅可比矩阵的行列式接近零，无法求逆。" << endl;
			exit(1);
		}
		vector<vector<double>> inv(3, vector<double>(3, 0.0));
		// 给定一个n×n的矩阵A，其伴随矩阵A∗定义为由A的代数余子式Cij组成的代数余子式矩阵C的转置。
		// https://blog.csdn.net/JiexianYao/article/details/140621381#:~:text=%E5%89%8D%E8%A8%80%EF%BC%9A%E6%83%B3%E8%A6%81%E5%AD%A6%E4%BC%9A%E3%80%8A%E7%BA%BF%E6%80%A7
		inv[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) / det;
		inv[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) / det;
		inv[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) / det;
		inv[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) / det;
		inv[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) / det;
		inv[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / det;
		inv[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) / det;
		inv[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) / det;
		inv[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) / det;
		return inv;
	}
private:
	double E;    // 弹性模量，单位 Pa
	double u;    // 泊松比，无量纲
	Eigen::Matrix<double, 6, 6> D;   // 偏应变矩阵 D
	Eigen::Matrix<double, 6, 6> D_dev;   // 偏应变矩阵 D
	Eigen::Matrix<double, 6, 6> D_dil;   // 体应变矩阵 D
};