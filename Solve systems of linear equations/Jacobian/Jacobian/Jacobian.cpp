#include <iostream>
#include <fstream>
using namespace std;

double tolerance = 1e-9;   //���������
int maxIterations = 1000;  //���õ���������

void Initialize_Output()//�洢�ļ���ʼ��
{
	ofstream file_writer1("data_jacobi.txt", ios::out);
	ofstream file_writer2("data_GS.txt", ios::out);
	ofstream file_writer3("data_RE_jacobi.txt", ios::out);
	ofstream file_writer4("data_RE_GS.txt", ios::out);
}

void Output_J(double* x, int col)//�������jacobiÿ�������������Ž��ļ���
{
	fstream f;
	f.open("data_jacobi.txt", ios::out | ios::app);
	for (int i = 0; i < col; i++)
	{
		f << fixed << setprecision(10) << x[i] << ' ';
	}
	f << endl;
	f.close();
}