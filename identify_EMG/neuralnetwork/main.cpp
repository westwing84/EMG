/****************************
�j���[�����l�b�g���[�N�̎���
���t�f�[�^�����NN�ւ̓��͂̓t�@�C������ǂݍ��ށD
NN�̓��o�͌��C�w���C�f�q���C�w�K���͎��s���Ɏw��\�D
****************************/
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


//main�֐�
int main(void) {
	
	int layer, elenum, input, output;
	double epsilon;
	double error;	//�덷
	int command;
	int teaching_data_size = 0;
	int learning_times;
	double error_min = 0.001;
	string filename_t_in;
	string filename_t_out;

	printf("���t�f�[�^�t�@�C����(����)����͂��Ă�������: ");
	cin >> filename_t_in;
	printf("���t�f�[�^�t�@�C����(�o��)����͂��Ă�������: ");
	cin >> filename_t_out;

	//���t�f�[�^�̃f�[�^�����J�E���g
	ifstream ifs(filename_t_in);
	if (!ifs) {
		printf("���t�f�[�^�t�@�C�����J���܂���ł����D\n");
		return 0;
	}
	string buf;
	while (getline(ifs, buf)) {
		teaching_data_size++;
	}
	ifs.close();

	//�e�p�����[�^���L�[�{�[�h�������
	printf("���t�f�[�^���j���[�����l�b�g���[�N�Ɋw�K�����܂��D\n�w�K������͂��Ă�������: ");
	scanf_s("%lf", &epsilon);
	printf("NN�ւ̓��͐�: ");
	scanf_s("%d", &input);
	printf("NN����̏o�͐�: ");
	scanf_s("%d", &output);
	printf("�j���[�����l�b�g���[�N�̑w��: ");
	scanf_s("%d", &layer);
	printf("�e�w�̑f�q��: ");
	scanf_s("%d", &elenum);
	printf("�w�K�񐔂̏��: ");
	scanf_s("%d", &learning_times);

	//���t�f�[�^���i�[����z��
	vector<Tdata> teaching_data(teaching_data_size, Tdata(input, output));

	//���t�f�[�^�t�@�C���I�[�v��
	ifstream ifs_t_in(filename_t_in);
	ifstream ifs_t_out(filename_t_out);
	if ((!ifs_t_in) || (!ifs_t_out)) {
		printf("���t�f�[�^�t�@�C�����J���܂���ł����D\n");
		return 0;
	}

	//���t�f�[�^��teaching_data�ɓǂݍ���
	string str;
	for (int i = 0; getline(ifs_t_in, str); i++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int j = 0; getline(stream, tmp, ','); j++) {
			teaching_data[i].input[j] = atof(tmp.c_str());
		}
	}
	for (int i = 0; getline(ifs_t_out, str); i++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int j = 0; getline(stream, tmp, ','); j++) {
			teaching_data[i].output[j] = atof(tmp.c_str());
		}
	}
	ifs_t_in.close();
	ifs_t_out.close();

	//�f�[�^�̗̈�m��

	//�d��
	//omega[i][j][k]: i�w�ڂ�j�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̎}�̏d�݁D
	//�e�v�f���́Comega[layer][elenum+1][elenum]�Delenum+1�Ƃ��Ă���̂̓o�C�A�X���܂�ł��邽�߁D
	//0�w�ڂ͓��͑w�D�܂��Comega[i][0][k]�̓o�C�A�X�ł���D
	vector<vector<vector<double>>> omega(layer, vector<vector<double>>(elenum + 1, vector<double>(elenum)));

	//�e�j���[�����ւ̓��́D�o�C�A�X�������D
	//x[i][j][k]: i�w�ڂ�(j+1)�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̓��́D
	//�e�v�f���́Cx[layer][elenum][elenum]�D
	vector<vector<vector<double>>> x(layer, vector<vector<double>>(elenum, vector<double>(elenum)));

	//�e�j���[��������̏o��
	//u[i][j]: (i+1)�w�ڂ�(j+1)�Ԗڂ̃j���[�����̏o�́D
	//�e�v�f���́Cu[layer-1][elenum]�D
	vector<vector<double>> u(layer - 1, vector<double>(elenum));

	//�t�`�d
	//dLdx[i][j][k]: (i+2)�w�ڂ�(j+1)�Ԗڂ̃j���[��������(k+1)�Ԗڂ̃j���[�����ւ̋t�`�d�o�́D
	//�e�v�f���́CdLdx[layer-1][elenum][elenum]�D
	vector<vector<vector<double>>> dLdx(layer - 1, vector<vector<double>>(elenum,vector<double>(elenum)));

	//NN�̏o��
	vector<double> y(output);

	//�d�݂̏����l��-1�`1�̗����ɂ�茈��
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1) * 2 - 1;
			}
		}
	}
	
	//���t�f�[�^�̏��Ԃ��V���b�t��
	shuffle(teaching_data);
	
	//�w�K
	printf("���t�f�[�^���w�K���Ă��܂��D\n");
	for (int i = 0; i < learning_times; i++) {
		error = 0;
		for (int j = 0; j < teaching_data_size; j++) {
			error += learning(omega, x, u, dLdx, teaching_data[j], y, input, output, layer, elenum, epsilon);
		}
		error /= teaching_data_size;
		if (i % 10 == 0) printf("%lf\n", error);
		if (error < error_min) break;
	}
	printf("�w�K���������܂����D\n");

	//NN�ւ̃f�[�^����
	command = 2;
	ifstream ifs_nt_in, ifs_ans_out;
	ofstream ofs_nt_out;
	string filename_in, filename_out, filename_ans;
	while (command != 0) {
		printf("0: �I���C1: �f�[�^����\n�R�}���h����͂��Ă�������: ");
		scanf_s("%d", &command);
		switch (command)
		{
		case 0:
			printf("�I�����܂��D\n");
			break;

		case 1:
		{
			printf("�j���[�����l�b�g���[�N�ւ̓��͂��s���܂��D���̓t�@�C��������͂��Ă�������: ");
			cin >> filename_in;
			printf("�o�̓t�@�C��������͂��Ă�������: ");
			cin >> filename_out;
			printf("�o�͂̐����f�[�^�̃t�@�C��������͂��Ă�������: ");
			cin >> filename_ans;

			//���̓f�[�^�̃f�[�^�����J�E���g
			int nteaching_data_size = 0;
			ifs_nt_in.open(filename_in, ios::in);
			if (!ifs_nt_in) {
				printf("���̓f�[�^�t�@�C�����J���܂���ł����D\n");
				continue;
			}
			string buf;
			while (getline(ifs_nt_in, buf)) {
				nteaching_data_size++;
			}
			ifs_nt_in.close();

			//���o�̓f�[�^�̗̈���m��
			vector<Tdata> nteaching_data(nteaching_data_size, Tdata(input, output));
			vector<Tdata> ans_data(nteaching_data_size,Tdata(input, output));


			//���̓f�[�^��ǂݍ���
			ifs_nt_in.open(filename_in, ios::in);
			for (int i = 0; getline(ifs_nt_in, str); i++) {
				string tmp;
				stringstream stream;
				stream << str;
				for (int j = 0; getline(stream, tmp, ','); j++) {
					nteaching_data[i].input[j] = atof(tmp.c_str());
				}
			}

			//�����f�[�^��ǂݍ���
			ifs_ans_out.open(filename_ans, ios::in);
			if (!ifs_ans_out) {
				printf("�����f�[�^�t�@�C�����J���܂���ł����D\n");
				continue;
			}
			for (int i = 0; getline(ifs_ans_out, str); i++) {
				string tmp;
				stringstream stream;
				stream << str;
				for (int j = 0; getline(stream, tmp, ','); j++) {
					ans_data[i].output[j] = atof(tmp.c_str());
				}
			}

			//NN�֓���
			for (int i = 0; i < nteaching_data_size; i++) {
				transmission(omega, x, u, nteaching_data[i].input, nteaching_data[i].output, input, output, layer, elenum);
			}

			//�t�@�C���փf�[�^���o��
			ofs_nt_out.open(filename_out, ios::out);
			for (int i = 0; i < nteaching_data_size; i++) {
				for (int j = 0; j < output + 1; j++) {
					if (j == 0) ofs_nt_out << i << ",";
					else ofs_nt_out << nteaching_data[i].output[j - 1] << ",";
				}
				ofs_nt_out << endl;
			}

			cout << "�o�̓f�[�^��" << filename_out << "�ɏo�͂��܂����D" << endl;

			//���ʗ��̎Z�o
			double id_rate;
			id_rate = calc_identification_rate(nteaching_data, ans_data, nteaching_data_size, output);
			cout << "���ʗ���" << id_rate << "%�ł��D" << endl;

			ifs_nt_in.close();
			ifs_ans_out.close();
			ofs_nt_out.close();

			break;
		}
		default:
			printf("������x���͂��Ă��������D\n");
			break;
		}
	}

	return 0;
}