#include "My_C3D8.h"

int main() {
	My_C3D8 c3d8;
	// ��ȡ inp �ļ�����ýڵ�������Ϣ Nodes ����Ԫ��Ϣ Elements
	c3d8.Init();
	MeshData data = c3d8.Readmesh("DataFile.inp");
	double E = 210000;
	double u = 0.3;
	// �������� Forces = [�����ڵ�, �������� (1,2,3 �ֱ���� x,y,z), ������С]
	vector<vector<double>> Forces{
		{5,   2, -100},
		{6,   2, -100},
		{56,  2, -100},
		{57,  2, -100},
		{58,  2, -100},
		{59,  2, -100},
		{60,  2, -100},
		{61,  2, -100},
		{62,  2, -100}
	};
	// Լ���ڵ�ı��
	std::vector<int> ConNumber = {
		9, 12, 13, 14, 116, 117, 118, 119, 120, 121, 122, 141, 142, 143, 144, 145,
		146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
		516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531,
		532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
		548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
		564
	};
	// Լ������ Constraints = [ǿ��λ�ƽڵ�, ǿ��λ�Ʒ��� (1,2,3 �ֱ���� x,y,z), ǿ��λ�ƴ�С]
	std::vector<std::vector<double>> Constraints;
	for (size_t i = 0; i < ConNumber.size(); ++i) {
		int node = ConNumber[i];
		Constraints.push_back({ static_cast<double>(node), 1.0, 0.0 });
		Constraints.push_back({ static_cast<double>(node), 2.0, 0.0 });
		Constraints.push_back({ static_cast<double>(node), 3.0, 0.0 });
	}
	vector<double> U;
	c3d8.StaticsSolver(E, u, Forces, Constraints, data.Nodes, data.Elements,U);
	vector<Eigen::VectorXd> NodeStrain;
	vector<Eigen::VectorXd> NodeStress;
	vector<Eigen::VectorXd> GaussStrain;
	vector<Eigen::VectorXd> GaussStress;
	c3d8.CalculateStrainAndStress(U, data.Nodes, data.Elements, NodeStrain, NodeStress, GaussStrain, GaussStress);
	c3d8.OutputResults("Results.txt", data.Nodes, data.Elements, U);
}