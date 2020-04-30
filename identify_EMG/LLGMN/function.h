//関数のプロトタイプ宣言

#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include "tdata_class.h"

using namespace std;

static const int BATCH = 1, SEQUENTIAL = 2;	//BATCH: 一括学習，SEQUENTIAL: 逐次学習

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
);

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
);

//重みの初期化
void init_weight(vector<vector<vector<double>>>& weight);

//教師データの順番をシャッフル
void shuffle(vector<Tdata> vec);