/****************************************
LLGMN�̎���
main�֐�����
******************************************/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include "tdata_class.h"
#include "function.h"

using namespace std;


int main() {

	//�ϐ��錾
	int teaching_data_size = 0;			//���t�f�[�^�̐�
	int input_size;						//���͂̌�(����)
	int k_class;						//�N���X(=�o�͂̌�)
	int component;						//�R���|�[�l���g
	double learning_rate;				//�w�K��
	int learning_times;					//�ő�w�K��
	double efunc_min = 0.01;			//�]���֐��̎�������l
	int non_linear_input_size;			//����`�ϊ���̓��͂̌�
	int mode;							//�ꊇ�w�K�������w�K����I��(BATCH: �ꊇ�w�K�CSEQUENTIAL: �����w�K)
	string filename_t_in, filename_t_out;	//���t�f�[�^�̃t�@�C����

	//���t�f�[�^�t�@�C�����C���͎����C�N���X���C�R���|�[�l���g���C�w�K���C�ő�w�K�񐔂����
	cout << "���t�f�[�^(����)�̃t�@�C��������͂��Ă�������: ";
	cin >> filename_t_in;
	cout << "���t�f�[�^(�o��)�̃t�@�C��������͂��Ă�������: ";
	cin >> filename_t_out;
	cout << "���̓f�[�^�̎���������͂��Ă�����: " ;
	cin >> input_size;
	cout << "�N���X������͂��Ă�����: ";
	cin >> k_class;
	cout << "�R���|�[�l���g������͂��Ă�����: ";
	cin >> component;
	cout << "�w�K������͂��Ă�����: ";
	cin >> learning_rate;
	cout << "�ő�w�K�񐔂���͂��Ă�����: ";
	cin >> learning_times;
	non_linear_input_size = 1 + input_size * (input_size + 3) / 2;
	cout << "�w�K���@����͂��Ă��������D1: �ꊇ�w�K�C2: �����w�K: ";
	cin >> mode;

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

	vector<Tdata> teaching_data(teaching_data_size, Tdata(input_size, k_class));	//���t�f�[�^

	//���t�f�[�^�t�@�C���I�[�v������ѓǂݍ���
	//���̓f�[�^
	ifstream ifs_t_in(filename_t_in);
	if (!ifs_t_in) {
		printf("���t�f�[�^�t�@�C��(����)���J���܂���ł����D\n");
		return 0;
	}
	string str;
	for (int n = 0; getline(ifs_t_in, str); n++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int d = 0; getline(stream, tmp, ','); d++) {
			teaching_data[n].input[d] = atof(tmp.c_str());
		}
	}
	ifs_t_in.close();

	//�o�̓f�[�^
	ifstream ifs_t_out(filename_t_out);
	if (!ifs_t_out) {
		printf("���t�f�[�^�t�@�C��(�o��)���J���܂���ł����D\n");
		return 0;
	}
	for (int n = 0; getline(ifs_t_out, str); n++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int d = 0; getline(stream, tmp, ','); d++) {
			teaching_data[n].output[d] = atof(tmp.c_str());
		}
	}
	ifs_t_out.close();

	//���t�f�[�^�̏��Ԃ��V���b�t��
	shuffle(teaching_data);


	/***********************************
	��1�w�����2�w�ւ̃u�����`�ɑ΂���d�݁D
	weight[h][k][m]: 1�w�ڂ�h-1�Ԗڂ̃m�[�h����2�w�ڂ̃N���Xk-1�C�R���|�[�l���gm-1�̃m�[�h�ւ̃u�����`�ɑ΂���d�݁D
	�e�v�f���͈̔͂͂��ꂼ��Ch: 0�`non_linear_input_size-1, k: 0�`k_class-1�Cm: 0�`component-1�D
	***********************************/
	vector<vector<vector<double>>> weight(non_linear_input_size, vector<vector<double>>(k_class, vector<double>(component)));

	/***********************************
	�e�m�[�h�ɑ΂�����́D�_�����ɂ�����I�ɑ����D
	In[l][k][m]:	l-1�w�ڂ̃N���Xk-1�C�R���|�[�l���gm-1�̃m�[�h�ւ̓���(2�w�ڂ̏ꍇ)�D
					l-1�w�ڂ�k-1�Ԗڂ̃m�[�h�ւ̓���(1,3�w�ڂ̏ꍇ)�Dm=0�̏ꍇ�̂݃f�[�^�����D
	l: 0�`2, k: 0�`max(non_linear_input_size,k_class)-1, m: 0�`component-1.
	***********************************/
	vector<vector<vector<double>>> In(3, vector<vector<double>>(max(non_linear_input_size, k_class), vector<double>(component)));

	/***********************************
	�e�m�[�h�ɑ΂���o�́D�_�����ɂ�����O�ɑ����D
	Out[n][l][k][m]:	n-1�ڂ̓��̓f�[�^�ɑ΂���o�́D
						l-1�w�ڂ̃N���Xk-1�C�R���|�[�l���gm-1�̃m�[�h����̏o��(2�w�ڂ̏ꍇ)�D
						l-1�w�ڂ�k-1�Ԗڂ̃m�[�h����̏o��(1,3�w�ڂ̏ꍇ)�Dm=0�̏ꍇ�̂݃f�[�^�����D
	n: 0�`teaching_data_size-1, l: 0�`2, k: 0�`max(non_linear_input_size,k_class)-1, m: 0�`component-1.
	***********************************/
	vector<vector<vector<vector<double>>>> Out(teaching_data_size, vector<vector<vector<double>>>(3, vector<vector<double>>(max(non_linear_input_size, k_class), vector<double>(component))));

	/***********************************
	���͂����`�ϊ�������̃f�[�^�D
	non_linear_input[n][h]: n-1�ڂ̓��̓f�[�^�ɑ΂���h-1�Ԗڂ̃f�[�^�D
	n: 0�`teaching_data_size-1, h: 0�`non_linear_input_size-1.
	***********************************/
	vector<vector<double>> non_linear_input(teaching_data_size, vector<double>(non_linear_input_size));

	/***********************************
	NN����̏o�́D
	output[n][k]: n-1�ڂ̓��̓f�[�^�ɑ΂���k-1�Ԗڂ̏o�́D
	n: 0�`teaching_data_size-1, k: 0�`k_class.
	***********************************/
	vector<vector<double>> output(teaching_data_size, vector<double>(k_class));


	//�d�݂̏�����
	init_weight(weight);

	//�w�K
	cout << "���t�f�[�^���w�K���Ă��܂��D" << endl;
	learning(teaching_data, weight, In, Out, non_linear_input, output, teaching_data_size, input_size, k_class, component, learning_rate, learning_times, efunc_min, mode);
	cout << "�w�K���I�����܂����D" << endl;

	//NN�ւ̖��w�K�f�[�^���͂܂��̓v���O�����I��
	int command = 0;
	string filename_nt_in, filename_nt_out;

	while (command != 2) {
		cout << "1: �f�[�^���� 2: �I��\n�R�}���h����͂��Ă�������: ";
		cin >> command;

		switch (command)
		{
		case 1:
		{
			cout << "NN�֓��͂���f�[�^�̃t�@�C��������͂��Ă�������: ";
			cin >> filename_nt_in;

			//���w�K�f�[�^�̓ǂݍ���
			//�s���J�E���g
			int non_teaching_data_size = 0;	//���w�K�f�[�^�̌�
			ifstream ifs(filename_nt_in);
			if (!ifs) {
				printf("���w�K�f�[�^�t�@�C�����J���܂���ł����D\n");
			}
			while (getline(ifs, buf)) {
				non_teaching_data_size++;
			}
			ifs.close();

			vector<Tdata> non_teaching_data(non_teaching_data_size, Tdata(input_size, k_class));	//���w�K�f�[�^

			//�ǂݍ���
			ifstream ifs_nt_in(filename_nt_in);
			for (int n = 0; getline(ifs_nt_in, str); n++) {
				string tmp;
				stringstream stream;
				stream << str;
				for (int d = 0; getline(stream, tmp, ','); d++) {
					non_teaching_data[n].input[d] = atof(tmp.c_str());
				}
			}
			ifs_nt_in.close();

			//���w�K�f�[�^��NN�ɓ���
			cout << filename_nt_in << "�̃f�[�^��NN�ɓ��͂��܂��D" << endl;
			vector<vector<vector<double>>> Out_nt(3, vector<vector<double>>(max(non_linear_input_size, k_class), vector<double>(component)));
			vector<double> non_linear_input_nt(non_linear_input_size);
			for (int ntdata_num = 0; ntdata_num < non_teaching_data_size; ntdata_num++) {
				forwardprop(weight, In, Out_nt, non_teaching_data[ntdata_num].input, non_linear_input_nt, non_teaching_data[ntdata_num].output, input_size, k_class, component);
			}

			//���ʌ��ʂ��t�@�C���ɏo��
			cout << "�o�̓t�@�C��������͂��Ă�������: ";
			cin >> filename_nt_out;
			ofstream ofs(filename_nt_out);
			for (int n = 0; n < non_teaching_data_size; n++) {
				for (int k = 0; k < k_class; k++) {
					ofs << non_teaching_data[n].output[k] << ",";
				}
				ofs << "\n";
			}
			ofs.close();
			cout << "���ʌ��ʂ�" << filename_nt_out << "�ɏo�͂��܂����D" << endl;
			break;
		}
			
		case 2:
			cout << "�I�����܂��D" << endl;
			break;

		default:
			cout << "������x���͂��Ă��������D" << endl;
			break;
		}
	}

	return 0;
}