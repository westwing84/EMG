/****************************
ニューラルネットワークの実装
教師データおよびNNへの入力はファイルから読み込む．
NNの入出力個数，層数，素子数，学習率は実行時に指定可能．
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


//main関数
int main(void) {
	
	int layer, elenum, input, output;
	double epsilon;
	double error;	//誤差
	int command;
	int teaching_data_size = 0;
	int learning_times;
	double error_min = 0.001;
	string filename_t_in;
	string filename_t_out;

	printf("教師データファイル名(入力)を入力してください: ");
	cin >> filename_t_in;
	printf("教師データファイル名(出力)を入力してください: ");
	cin >> filename_t_out;

	//教師データのデータ数をカウント
	ifstream ifs(filename_t_in);
	if (!ifs) {
		printf("教師データファイルを開けませんでした．\n");
		return 0;
	}
	string buf;
	while (getline(ifs, buf)) {
		teaching_data_size++;
	}
	ifs.close();

	//各パラメータをキーボードから入力
	printf("教師データをニューラルネットワークに学習させます．\n学習率を入力してください: ");
	scanf_s("%lf", &epsilon);
	printf("NNへの入力数: ");
	scanf_s("%d", &input);
	printf("NNからの出力数: ");
	scanf_s("%d", &output);
	printf("ニューラルネットワークの層数: ");
	scanf_s("%d", &layer);
	printf("各層の素子数: ");
	scanf_s("%d", &elenum);
	printf("学習回数の上限: ");
	scanf_s("%d", &learning_times);

	//教師データを格納する配列
	vector<Tdata> teaching_data(teaching_data_size, Tdata(input, output));

	//教師データファイルオープン
	ifstream ifs_t_in(filename_t_in);
	ifstream ifs_t_out(filename_t_out);
	if ((!ifs_t_in) || (!ifs_t_out)) {
		printf("教師データファイルを開けませんでした．\n");
		return 0;
	}

	//教師データをteaching_dataに読み込み
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

	//データの領域確保

	//重み
	//omega[i][j][k]: i層目のj番目のニューロンから(i+1)層目の(k+1)番目のニューロンへの枝の重み．
	//各要素数は，omega[layer][elenum+1][elenum]．elenum+1としているのはバイアスも含んでいるため．
	//0層目は入力層．また，omega[i][0][k]はバイアスである．
	vector<vector<vector<double>>> omega(layer, vector<vector<double>>(elenum + 1, vector<double>(elenum)));

	//各ニューロンへの入力．バイアスを除く．
	//x[i][j][k]: i層目の(j+1)番目のニューロンから(i+1)層目の(k+1)番目のニューロンへの入力．
	//各要素数は，x[layer][elenum][elenum]．
	vector<vector<vector<double>>> x(layer, vector<vector<double>>(elenum, vector<double>(elenum)));

	//各ニューロンからの出力
	//u[i][j]: (i+1)層目の(j+1)番目のニューロンの出力．
	//各要素数は，u[layer-1][elenum]．
	vector<vector<double>> u(layer - 1, vector<double>(elenum));

	//逆伝播
	//dLdx[i][j][k]: (i+2)層目の(j+1)番目のニューロンから(k+1)番目のニューロンへの逆伝播出力．
	//各要素数は，dLdx[layer-1][elenum][elenum]．
	vector<vector<vector<double>>> dLdx(layer - 1, vector<vector<double>>(elenum,vector<double>(elenum)));

	//NNの出力
	vector<double> y(output);

	//重みの初期値を-1～1の乱数により決定
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1) * 2 - 1;
			}
		}
	}
	
	//教師データの順番をシャッフル
	shuffle(teaching_data);
	
	//学習
	printf("教師データを学習しています．\n");
	for (int i = 0; i < learning_times; i++) {
		error = 0;
		for (int j = 0; j < teaching_data_size; j++) {
			error += learning(omega, x, u, dLdx, teaching_data[j], y, input, output, layer, elenum, epsilon);
		}
		error /= teaching_data_size;
		if (i % 10 == 0) printf("%lf\n", error);
		if (error < error_min) break;
	}
	printf("学習が完了しました．\n");

	//NNへのデータ入力
	command = 2;
	ifstream ifs_nt_in, ifs_ans_out;
	ofstream ofs_nt_out;
	string filename_in, filename_out, filename_ans;
	while (command != 0) {
		printf("0: 終了，1: データ入力\nコマンドを入力してください: ");
		scanf_s("%d", &command);
		switch (command)
		{
		case 0:
			printf("終了します．\n");
			break;

		case 1:
		{
			printf("ニューラルネットワークへの入力を行います．入力ファイル名を入力してください: ");
			cin >> filename_in;
			printf("出力ファイル名を入力してください: ");
			cin >> filename_out;
			printf("出力の正解データのファイル名を入力してください: ");
			cin >> filename_ans;

			//入力データのデータ数をカウント
			int nteaching_data_size = 0;
			ifs_nt_in.open(filename_in, ios::in);
			if (!ifs_nt_in) {
				printf("入力データファイルを開けませんでした．\n");
				continue;
			}
			string buf;
			while (getline(ifs_nt_in, buf)) {
				nteaching_data_size++;
			}
			ifs_nt_in.close();

			//入出力データの領域を確保
			vector<Tdata> nteaching_data(nteaching_data_size, Tdata(input, output));
			vector<Tdata> ans_data(nteaching_data_size,Tdata(input, output));


			//入力データを読み込み
			ifs_nt_in.open(filename_in, ios::in);
			for (int i = 0; getline(ifs_nt_in, str); i++) {
				string tmp;
				stringstream stream;
				stream << str;
				for (int j = 0; getline(stream, tmp, ','); j++) {
					nteaching_data[i].input[j] = atof(tmp.c_str());
				}
			}

			//正解データを読み込み
			ifs_ans_out.open(filename_ans, ios::in);
			if (!ifs_ans_out) {
				printf("正解データファイルを開けませんでした．\n");
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

			//NNへ入力
			for (int i = 0; i < nteaching_data_size; i++) {
				transmission(omega, x, u, nteaching_data[i].input, nteaching_data[i].output, input, output, layer, elenum);
			}

			//ファイルへデータを出力
			ofs_nt_out.open(filename_out, ios::out);
			for (int i = 0; i < nteaching_data_size; i++) {
				for (int j = 0; j < output + 1; j++) {
					if (j == 0) ofs_nt_out << i << ",";
					else ofs_nt_out << nteaching_data[i].output[j - 1] << ",";
				}
				ofs_nt_out << endl;
			}

			cout << "出力データを" << filename_out << "に出力しました．" << endl;

			//識別率の算出
			double id_rate;
			id_rate = calc_identification_rate(nteaching_data, ans_data, nteaching_data_size, output);
			cout << "識別率は" << id_rate << "%です．" << endl;

			ifs_nt_in.close();
			ifs_ans_out.close();
			ofs_nt_out.close();

			break;
		}
		default:
			printf("もう一度入力してください．\n");
			break;
		}
	}

	return 0;
}