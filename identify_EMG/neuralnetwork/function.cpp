#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include "function.h"
#include "tdata_class.h"
using namespace std;

//�V�O���C�h�֐�
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}


/**********************************
���t�f�[�^���w�K�����C�d��omega�����肷��D
�߂�l�͋��t�f�[�^�Əo�͂̌덷�D
***********************************/
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
) {
	double b = 1;				//�o�C�A�X�ɑ΂������
	double error, error_out;	//�덷
	vector<double> dLdx_sum(elenum, 0);	//�t�`�d�̘a

	//���`�d�ɂ�苳�t����t_in�ɑ΂���o��y���v�Z����
	transmission(omega, x, u, teaching_data.input, y, input, output, layer, elenum);

	//�o�͂̌덷�̌v�Z
	error_out = 0;
	for (int i = 0; i < output; i++) {
		error_out += pow(y[i] - teaching_data.output[i], 2);
	}

	//�덷�t�`�d�@�ɂ��d�݂̍X�V
	//�o�͑w
	for (int i = 0; i < elenum + 1; i++) {
		for (int j = 0; j < output; j++) {
			if (i == 0) error = 2 * b * y[j] * (y[j] - teaching_data.output[j]) * (1 - y[j]);
			else error = 2 * x[layer - 1][i - 1][j] * y[j] * (y[j] - teaching_data.output[j]) * (1 - y[j]);
			omega[layer - 1][i][j] -= epsilon * error;
			if (i > 0) {
				dLdx[layer - 2][i - 1][j] = 2 * omega[layer - 1][i][j] * y[j] * (y[j] - teaching_data.output[j]) * (1 - y[j]);
				dLdx_sum[i - 1] += dLdx[layer - 2][i - 1][j];
			}
		}
	}

	//���ԑw
	for (int i = layer - 2; i > 0; i--) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				if (j == 0) error = b * u[i][k] * (1 - u[i][k]) * dLdx_sum[k];
				else error = x[i][j - 1][k] * u[i][k] * (1 - u[i][k]) * dLdx_sum[k];
				omega[i][j][k] -= epsilon * error;
				if (j > 0) {
					dLdx[i - 1][j - 1][k] = omega[i][j][k] * u[i][k] * (1 - u[i][k]) * dLdx_sum[k];
				}
			}
		}
		for (int j = 0; j < elenum; j++) {
			dLdx_sum[j] = 0;
		}
		for (int j = 0; j < elenum; j++) {
			for (int k = 0; k < elenum; k++) {
				dLdx_sum[j] += dLdx[i - 1][j][k];
			}
		}
	}

	//���͑w
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) error = b * u[0][j] * (1 - u[0][j]) * dLdx_sum[j];
			else error = x[0][i - 1][j] * u[0][j] * (1 - u[0][j]) * dLdx_sum[j];
			omega[0][i][j] -= epsilon * error;
		}
	}
	return error_out;
}


//���`�d�ɂ��NN�̏o��y�𓾂�֐�
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
) {
	double b = 1;	//�o�C�A�X
	//�j���[��������̏o��u����ёS�̂̏o��y�̏�����
	for (int i = 0; i < layer - 1; i++) {
		for (int j = 0; j < elenum; j++) {
			u[i][j] = 0;
		}
	}
	for (int i = 0; i < output; i++) {
		y[i] = 0;
	}

	//���͑w�����1�w�ւ̓`�B
	for (int i = 0; i < input; i++) {
		for (int j = 0; j < elenum; j++) {
			x[0][i][j] = in[i];
		}
	}
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) u[0][j] += b * omega[0][i][j];
			else u[0][j] += in[i - 1] * omega[0][i][j];
		}
	}

	//���ԑw�̓`�B(3�w�ȏ�̏ꍇ�̂�)
	for (int i = 1; i < layer - 1; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			if (j > 0) u[i - 1][j - 1] = sigmoid(u[i - 1][j - 1]);
			for (int k = 0; k < elenum; k++) {
				if (j == 0) u[i][k] += b * omega[i][j][k];
				else {
					x[i][j - 1][k] = u[i - 1][j - 1];
					u[i][k] += x[i][j - 1][k] * omega[i][j][k];
				}
			}
		}
	}

	//��(layer-1)�w����o�͑w(��layer�w)�ւ̓`�B
	for (int j = 0; j < elenum + 1; j++) {
		if (j > 0) u[layer - 2][j - 1] = sigmoid(u[layer - 2][j - 1]);
		for (int k = 0; k < output; k++) {
			if (j == 0) y[k] += b * omega[layer - 1][j][k];
			else {
				x[layer - 1][j - 1][k] = u[layer - 2][j - 1];
				y[k] += x[layer - 1][j - 1][k] * omega[layer - 1][j][k];
			}
		}
	}
	for (int k = 0; k < output; k++) {
		y[k] = sigmoid(y[k]);
	}
}

//���t�f�[�^�̏��Ԃ��V���b�t��
void shuffle(vector<Tdata> vec) {
	mt19937 mt;
	shuffle(vec.begin(), vec.end(), mt);
}