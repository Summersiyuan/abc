#pragma once
#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include "get_info.h"
#include "lib_io.h"

#define F 100 //所需物理服务器的最大值

extern int predict_server_all;//预测虚拟机（CPU或MEM）总容量
extern int number_server;//需要物理服务器的数目
extern int output_i[][N];//[服务器]、[虚拟机型号]、虚拟机个数
extern void put(unsigned int core_num, unsigned int RAM, int flavor[][3], unsigned int opti_type);

int my_pow(int x, int y);
double my_ceil(double x);

#endif


