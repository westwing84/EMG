//関数のプロトタイプ宣言

#pragma once
#include <vector>
#include "tdata_class.h"

double sigmoid(double s);
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
