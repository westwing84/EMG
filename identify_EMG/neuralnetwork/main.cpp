/****************************
�j���[�����l�b�g���[�N�̎���
���t�f�[�^�����NN�ւ̓��͂̓t�@�C������ǂݍ��ށD
NN�̓��o�͌��C�w���C�f�q���C�w�K���͎��s���Ɏw��\�D
****************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


//�֐��̃v���g�^�C�v�錾
double sigmoid(double s);
double learning(
	double t_in[],		//���͋��t�f�[�^
	double t_out[],		//�o�͋��t�f�[�^
	double y[],			//NN�̏o��
	int input,			//NN�̓��͂̌�
	int output,			//NN�̏o�͂̌�
	int layer,			//NN�̑w��
	int elenum,			//�e�w�ɂ�����j���[�����̌�
	double epsilon		//�w�K��
);
void transmission(double in[], double y[], int input, int output, int layer, int elenum);
int get_rand(int min_val, int max_val);
void shuffle(double** array1, double** array2, int size);

//�ϐ��錾
static double*** omega;	//�d��
//omega[i][j][k]: i�w�ڂ�j�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̎}�̏d�݁D
//�e�v�f���́Comega[layer][elenum+1][elenum]�Delenum+1�Ƃ��Ă���̂̓o�C�A�X���܂�ł��邽�߁D
//0�w�ڂ͓��͑w�D�܂��Comega[i][0][k]�̓o�C�A�X�ł���D

static double*** x;		//�e�j���[�����ւ̓��́D�o�C�A�X�������D
//x[i][j][k]: i�w�ڂ�(j+1)�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̓��́D
//�e�v�f���́Cx[layer][elenum][elenum]�D

static double** u;		//�e�j���[��������̏o��
//u[i][j]: (i+1)�w�ڂ�(j+1)�Ԗڂ̃j���[�����̏o�́D
//�e�v�f���́Cu[layer-1][elenum]�D

static double** dLdx;	//�t�`�d
//dLdx[i][j]: (i+2)�w�ڂ�(j+1)�Ԗڂ̃j���[�����̋t�`�d�o�́D
//�e�v�f���́CdLdx[layer-1][elenum]�D

//main�֐�
int main(void) {
	
	int layer, elenum, input, output;
	double epsilon;
	double** dt_in, **dt_out;
	double* y;		//NN�̏o��
	double error;	//�덷
	int command;
	double** t_in;
	double** t_out;
	int teaching_data_size = 0;
	int learning_times;
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

	//���t�f�[�^�̗̈�m��
	t_in = (double**)calloc(teaching_data_size, sizeof(double));
	for (int i = 0; i < teaching_data_size; i++) {
		t_in[i] = (double*)calloc(input, sizeof(double));
	}
	t_out = (double**)calloc(teaching_data_size, sizeof(double));
	for (int i = 0; i < teaching_data_size; i++) {
		t_out[i] = (double*)calloc(output, sizeof(double));
	}

	//���t�f�[�^�t�@�C���I�[�v��
	ifstream ifs_t_in(filename_t_in);
	ifstream ifs_t_out(filename_t_out);
	if ((!ifs_t_in) || (!ifs_t_out)) {
		printf("���t�f�[�^�t�@�C�����J���܂���ł����D\n");
		return 0;
	}

	//���t�f�[�^��t_in��t_out�ɓǂݍ���
	string str;
	for (int i = 0; getline(ifs_t_in, str); i++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int j = 0; getline(stream, tmp, ','); j++) {
			t_in[i][j] = atof(tmp.c_str());
		}
	}
	for (int i = 0; getline(ifs_t_out, str); i++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int j = 0; getline(stream, tmp, ','); j++) {
			t_out[i][j] = atof(tmp.c_str());
		}
	}
	ifs_t_in.close();
	ifs_t_out.close();

	//omega�̃f�[�^�̈�m��
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum + 1, sizeof(double));
		for (int j = 0; j < elenum + 1; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//�̈�m��
	x = (double***)calloc(layer, sizeof(double));
	u = (double**)calloc(layer - 1, sizeof(double));
	dLdx = (double**)calloc(layer - 1, sizeof(double));
	y = (double*)calloc(output, sizeof(double));
	for (int i = 0; i < layer; i++) {
		x[i] = (double**)calloc(elenum, sizeof(double));
		if (i < layer - 1) u[i] = (double*)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			x[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}
	for (int i = 0; i < layer - 1; i++) {
		dLdx[i] = (double*)calloc(elenum, sizeof(double));
	}

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
	shuffle(t_in, t_out, teaching_data_size);
	
	//�w�K
	printf("���t�f�[�^���w�K���Ă��܂��D\n");
	for (int i = 0; i < learning_times; i++) {
		error = 0;
		for (int j = 0; j < teaching_data_size; j++) {
			error += learning(t_in[j], t_out[j], y, input, output, layer, elenum, epsilon);
		}
		error /= teaching_data_size;
		if (i % 10 == 0) printf("%lf\n", error);
		if (error < 1e-5) break;
	}
	printf("�w�K���������܂����D\n");

	//NN�ւ̃f�[�^����
	command = 2;
	ifstream ifs_nt_in;
	ofstream ofs_nt_out;
	string filename_in, filename_out;
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
			dt_in = (double**)calloc(nteaching_data_size, sizeof(double));
			dt_out = (double**)calloc(nteaching_data_size, sizeof(double));
			for (int i = 0; i < nteaching_data_size; i++) {
				dt_in[i] = (double*)calloc(input, sizeof(double));
				dt_out[i] = (double*)calloc(output, sizeof(double));
			}


			//���̓f�[�^��ǂݍ���
			ifs_nt_in.open(filename_in, ios::in);
			for (int i = 0; getline(ifs_nt_in, str); i++) {
				string tmp;
				stringstream stream;
				stream << str;
				for (int j = 0; getline(stream, tmp, ','); j++) {
					dt_in[i][j] = atof(tmp.c_str());
				}
			}

			//NN�֓���
			for (int i = 0; i < nteaching_data_size; i++) {
				transmission(dt_in[i], dt_out[i], input, output, layer, elenum);
			}

			//�t�@�C���փf�[�^���o��
			ofs_nt_out.open(filename_out, ios::out);
			for (int i = 0; i < nteaching_data_size; i++) {
				for (int j = 0; j < output; j++) {
					ofs_nt_out << dt_out[i][j] << ",";
				}
				ofs_nt_out << endl;
			}
			ifs_nt_in.close();
			ofs_nt_out.close();
			cout << "�o�̓f�[�^��" << filename_out << "�ɏo�͂��܂����D" << endl;
			
			//�̈���
			for (int i = 0; i < nteaching_data_size; i++) {
				free(dt_in[i]);
				free(dt_out[i]);
			}
			free(dt_in);
			free(dt_out);

			break;
		}
		default:
			printf("������x���͂��Ă��������D\n");
			break;
		}
	}
	

	//�̈�̉��
	for (int i = 0; i < teaching_data_size; i++) {
		free(t_in[i]);
		free(t_out[i]);
	}
	free(t_in);
	free(t_out);

	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(omega[i][j]);
		}
		free(omega[i]);
	}
	free(omega);
	free(y);

	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(x[i][j]);
		}
		free(x[i]);
		if (i < layer - 1) free(u[i]);
	}
	for (int i = 0; i < layer - 1; i++) {
		free(dLdx[i]);
	}
	free(x);
	free(u);
	free(dLdx);
	
	return 0;
}


//�V�O���C�h�֐�
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}


/**********************************
���t�f�[�^���w�K�����C�d��omega�����肷��D
�߂�l�͋��t�f�[�^�Əo�͂̌덷�D
***********************************/
double learning(
	double t_in[],		//���͋��t�f�[�^
	double t_out[],		//�o�͋��t�f�[�^
	double y[],			//NN�̏o��
	int input,			//NN�̓��͂̌�
	int output,			//NN�̏o�͂̌�
	int layer,			//NN�̑w��
	int elenum,			//�e�w�ɂ�����j���[�����̌�
	double epsilon		//�w�K��
) {
	double b = 1;				//�o�C�A�X�ɑ΂������
	double error, error_out;	//�덷
	double dLdx_sum = 0;		//�t�`�d�̘a

	//���`�d�ɂ�苳�t����t_in�ɑ΂���o��y���v�Z����
	transmission(t_in, y, input, output, layer, elenum);

	//�o�͂̌덷�̌v�Z
	error_out = 0;
	for (int i = 0; i < output; i++) {
		error_out += pow(y[i] - t_out[i], 2);
	}

	//�덷�t�`�d�@�ɂ��d�݂̍X�V
	//�o�͑w
	for (int i = 0; i < elenum + 1; i++) {
		for (int j = 0; j < output; j++) {
			if (i == 0) error = 2 * b * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			else error = 2 * x[layer - 1][i - 1][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			omega[layer - 1][i][j] -= epsilon * error;
			if (i > 0) dLdx[layer - 2][j] = 2 * omega[layer - 1][i][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
		}
	}

	//���ԑw
	for (int i = layer - 2; i > 0; i--) {
		for (int j = 0; j < elenum; j++) {
			dLdx_sum += dLdx[i][j];
		}
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				if (j == 0) error = b * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				else error = x[i][j - 1][k] * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				omega[i][j][k] -= epsilon * error;
				if (j > 0) dLdx[i - 1][j - 1] = omega[i][j][k] * u[i][k] * (1 - u[i][k]) * dLdx[i][k];
			}
		}
		dLdx_sum = 0;
	}

	//���͑w
	for (int k = 0; k < elenum; k++) {
		dLdx_sum += dLdx[0][k];
	}
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) error = b * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			else error = x[0][i - 1][j] * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			omega[0][i][j] -= epsilon * error;
		}
	}

	return error_out;
}


//���`�d�ɂ��NN�̏o��y�𓾂�֐�
void transmission(double in[], double y[], int input, int output, int layer, int elenum) {
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

//min_val����max_val-1�͈̔͂̐����̗����𐶐�����֐�
int get_rand(int min_val, int max_val) {
	return rand() % (max_val - min_val) + min_val;
}

//���t�f�[�^�̏��ԃV���b�t���p
void shuffle(double** array1, double** array2, int size) {
	for (int i = 0; i < size; i++) {
		int r = get_rand(i, size);
		double* tmp1 = array1[i], *tmp2 = array2[i];
		array1[i] = array1[r];
		array1[r] = tmp1;
		array2[i] = array2[r];
		array2[r] = tmp2;
	}
}