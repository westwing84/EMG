/****************************************
LLGMNの実装
関数部分
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

//順方向伝播
void forwardprop(
	vector<vector<vector<double>>>& weight,		//重み
	vector<vector<vector<double>>>& In,			//各ニューロンに対する入力
	vector<vector<vector<double>>>& Out,		//各ニューロンに対する出力
	vector<double>& input,						//NNへの入力
	vector<double>& non_linear_input,			//非線形変換後の入力
	vector<double>& output,		//NNからの出力
	int input_size,				//入力の個数(次元)
	int k_class,				//クラス(=出力の個数)
	int component				//コンポーネント
) {
	int non_linear_input_size = 1 + input_size * (input_size + 3) / 2;	//非線形変換後の入力の個数
	int count1 = 0, count2 = 0;

	//入力の非線形変換
	non_linear_input[0] = 1;
	for (int i = 0; i < input_size; i++) {
		non_linear_input[i + 1] = input[i];
	}
	for (int i = input_size + 1; i < non_linear_input_size; i++) {
		non_linear_input[i] = input[count1] * input[count2];
		count1++;
		if (count1 == input_size) {
			count2++;
			count1 = count2;
		}
	}


	//第1層の伝播
	for (int i = 0; i < non_linear_input_size; i++) {
		In[0][i][0] = non_linear_input[i];
		Out[0][i][0] = In[0][i][0];
	}


	//第2層の伝播
	for (int h = 0; h < non_linear_input_size; h++) {
		weight[h][k_class - 1][component - 1] = 0;	//第2層の一番最後のニューロンに対する重みは0とする
	}

	//2層目入力の計算
	for (int k = 0; k < k_class; k++) {
		for (int m = 0; m < component; m++) {
			In[1][k][m] = 0;
			for (int h = 0; h < non_linear_input_size; h++) {
				In[1][k][m] += Out[0][h][0] * weight[h][k][m];
			}
		}
	}

	//2層目出力の計算
	double sum = 0;
	for (int k = 0; k < k_class; k++) {
		for (int m = 0; m < component; m++) {
			sum += exp(In[1][k][m]);
		}
	}
	for (int k = 0; k < k_class; k++) {
		for (int m = 0; m < component; m++) {
			Out[1][k][m] = exp(In[1][k][m]) / sum;
		}
	}


	//第3層の伝播
	for (int k = 0; k < k_class; k++) {
		In[2][k][0] = 0;
		for (int m = 0; m < component; m++) {
			In[2][k][0] += Out[1][k][m];
		}
		output[k] = In[2][k][0];
	}
}


//学習
void learning(
	vector<Tdata> teaching_data,				//教師データ
	vector<vector<vector<double>>>& weight,		//重み
	vector<vector<vector<double>>>& In,			//各ニューロンに対する入力
	vector<vector<vector<vector<double>>>>& Out,//各ニューロンに対する出力
	vector<vector<double>>& non_linear_input,	//非線形変換後の入力
	vector<vector<double>>& output,				//NNからの出力
	int teaching_data_size,		//教師データの数
	int input_size,				//入力の個数(次元)
	int k_class,				//クラス(=出力の個数)
	int component,				//コンポーネント
	double learning_rate,		//学習率
	int learning_times,			//最大学習回数
	double efunc_min,			//評価関数の収束判定値
	int mode					//BATCH: 一括学習，SEQUENTIAL: 逐次学習
) {
	int non_linear_input_size = 1 + input_size * (input_size + 3) / 2;	//非線形変換後の入力の個数
	double evaluation_func;		//評価関数の値
	double diff_efunc;			//評価関数の微分
	double delta_weight;		//重みの更新値

	for (int times = 0; times < learning_times; times++) {
		//各教師データに対して順方向伝播を行い，出力を得る
		for (int tdata_num = 0; tdata_num < teaching_data_size; tdata_num++) {
			forwardprop(weight, In, Out[tdata_num], teaching_data[tdata_num].input, non_linear_input[tdata_num], output[tdata_num], input_size, k_class, component);
		}

		//評価関数の導出
		evaluation_func = 0;
		for (int n = 0; n < teaching_data_size; n++) {
			for (int k = 0; k < k_class; k++) {
				evaluation_func -= teaching_data[n].output[k] * log10(output[n][k]);
			}
		}
		evaluation_func = abs(evaluation_func) / teaching_data_size;
		if (times % 10 == 0) cout << evaluation_func << endl;
		if (abs(evaluation_func) < efunc_min) break;	//収束判定

		//重みの更新
		for (int h = 0; h < non_linear_input_size; h++) {
			for (int k = 0; k < k_class; k++) {
				for (int m = 0; m < component; m++) {
					if (mode == BATCH) {	//一括学習
						delta_weight = 0;
						for (int n = 0; n < teaching_data_size; n++) {
							diff_efunc = (output[n][k] - teaching_data[n].output[k]) * Out[n][1][k][m] / output[n][k] * non_linear_input[n][h];
							delta_weight += diff_efunc;
						}
						delta_weight *= -learning_rate;
						weight[h][k][m] += delta_weight;
					}
					else {	//逐次学習
						for (int n = 0; n < teaching_data_size; n++) {
							diff_efunc = (output[n][k] - teaching_data[n].output[k]) * Out[n][1][k][m] / output[n][k] * non_linear_input[n][h];
							delta_weight = diff_efunc;
							delta_weight *= -learning_rate;
							weight[h][k][m] += delta_weight;
						}
					}
				}
			}
		}
	}
}


//重みの初期化
void init_weight(vector<vector<vector<double>>>& weight) {
	random_device rnd;
	mt19937 mt(rnd());
	uniform_real_distribution<> rand12(-1.0, 1.0);	//-1～1の範囲の一様乱数

	for (int i = 0; i < (int)(weight.size()); i++) {
		for (int j = 0; j < (int)(weight[i].size()); j++) {
			for (int k = 0; k < (int)(weight[i][j].size()); k++) {
				weight[i][j][k] = rand12(mt);
			}
		}
	}
}


//教師データの順番をシャッフル
void shuffle(vector<Tdata> vec) {
	mt19937 mt;
	shuffle(vec.begin(), vec.end(), mt);
}