//教師データクラスの実装

#include <iostream>
#include <vector>
#include "tdata_class.h"

using namespace std;

//教師データクラスのコンストラクタ
Tdata::Tdata(int input_size, int output_size)
{
	input.resize(input_size, 0);
	output.resize(output_size, 0);
}