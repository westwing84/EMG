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

//シグモイド関数
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}


/**********************************
教師データを学習させ，重みomegaを決定する．
戻り値は教師データと出力の誤差．
***********************************/
double learning(
	vector<vector<vector<double>>>& omega,	//重み
	vector<vector<vector<double>>>& x,		//各ニューロンへの入力
	vector<vector<double>>& u,				//各ニューロンからの出力
	vector<vector<vector<double>>>& dLdx,	//各ニューロンからの逆伝播出力
	Tdata teaching_data,					//教師データ
	vector<double>& y,						//NNの出力
	int input,			//NNの入力の個数
	int output,			//NNの出力の個数
	int layer,			//NNの層数
	int elenum,			//各層におけるニューロンの個数
	double epsilon		//学習率
) {
	double b = 1;				//バイアスに対する入力
	double error, error_out;	//誤差
	vector<double> dLdx_sum(elenum, 0);	//逆伝播の和

	//順伝播により教師入力t_inに対する出力yを計算する
	transmission(omega, x, u, teaching_data.input, y, input, output, layer, elenum);

	//出力の誤差の計算
	error_out = 0;
	for (int i = 0; i < output; i++) {
		error_out += pow(y[i] - teaching_data.output[i], 2);
	}

	//誤差逆伝播法による重みの更新
	//出力層
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

	//中間層
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

	//入力層
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) error = b * u[0][j] * (1 - u[0][j]) * dLdx_sum[j];
			else error = x[0][i - 1][j] * u[0][j] * (1 - u[0][j]) * dLdx_sum[j];
			omega[0][i][j] -= epsilon * error;
		}
	}
	return error_out;
}


//順伝播によりNNの出力yを得る関数
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
	double b = 1;	//バイアス
	//ニューロンからの出力uおよび全体の出力yの初期化
	for (int i = 0; i < layer - 1; i++) {
		for (int j = 0; j < elenum; j++) {
			u[i][j] = 0;
		}
	}
	for (int i = 0; i < output; i++) {
		y[i] = 0;
	}

	//入力層から第1層への伝達
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

	//中間層の伝達(3層以上の場合のみ)
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

	//第(layer-1)層から出力層(第layer層)への伝達
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

//教師データの順番をシャッフル
void shuffle(vector<Tdata> vec) {
	mt19937 mt;
	shuffle(vec.begin(), vec.end(), mt);
}