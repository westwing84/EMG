//�֐��̃v���g�^�C�v�錾

#pragma once
#include <vector>
#include "tdata_class.h"

double sigmoid(double s);
double learning(
	vector<vector<vector<double>>>& omega,	//�d��
	vector<vector<vector<double>>>& x,		//�e�j���[�����ւ̓���
	vector<vector<double>>& u,				//�e�j���[��������̏o��
	vector<vector<vector<double>>>& dLdx,	//�e�j���[��������̋t�`�d�o��
	Tdata teaching_data,					//���t�f�[�^
	vector<double>& y,						//NN�̏o��
	int input,			//NN�̓��͂̌�
	int output,			//NN�̏o�͂̌�
	int layer,			//NN�̑w��
	int elenum,			//�e�w�ɂ�����j���[�����̌�
	double epsilon		//�w�K��
);
void transmission(
	vector<vector<vector<double>>>& omega,
	vector<vector<vector<double>>>& x,
	vector<vector<double>>& u,
	vector<double>& in,
	vector<double>& y,
	int input,
	int output,
	int layer,
	int elenum
);
void shuffle(vector<Tdata> vec);
double calc_identification_rate(vector<Tdata> nteaching_data, vector<Tdata> ans_data, int dtsize, int output_dtsize);
