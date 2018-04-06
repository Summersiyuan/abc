#ifndef __GET_INFO_H__
#define __GET_INFO_H__

#include "lib_io.h"

#define CPU 1
#define MEM 2 //待优化选项
#define N 25 //虚拟机类型总数
#define MAX_LINE_LEN 55000 //txt文本每行最大长度

#define MAX_TRAIN_DAYS 150
//extern char *a[];
extern char *t_flavor[]; //train文件的虚拟机类型数组
extern char *t_time_stamp[]; //train文件每个虚拟机对应的请求时间
extern unsigned int t_flavor_num[N]; //train文件里面每一类型虚拟机总量
extern unsigned int core_num; //物理服务器内核数量
extern unsigned int RAM; //物理服务器内存
extern unsigned int DISK;//物理服务器硬盘
extern char *i_flavor[]; //input文件的虚拟机类型
extern unsigned int i_flavorN;//预测虚拟机种类数
extern unsigned int i_flavor_core[]; //input文件的物理服务器内核数
extern unsigned int i_flavor_RAM[]; //input文件的物理服务器内存
extern char *predict_daystamp[]; //预测时间段，包含预测开始时间 预测结束时间
extern unsigned int predict_flavor[];//需要预测的虚拟机型号，从1开始
extern unsigned int  type_opti;//优化选项
extern int flavornum_day[MAX_TRAIN_DAYS][N]; //train文件每一天某一型号虚拟机使用数量 [天数][虚拟机类型]
extern int flavornum_day_cheng[MAX_TRAIN_DAYS][N]; //train文件每一天某一型号虚拟机使用数量 [天数][0-i_flavorN]  
extern int   flavor[][3];//定义虚拟机内核数和内存大小,1:内核数;2:内存大小
extern int t_gap, i_gap; //train文件的时间间隔 input文件的预测时间间隔
char *str_split(unsigned char seg_n, char * const buff, char * str_split, const char * split_type);
//字符串分割函数,返回字符串
unsigned int int_split(unsigned char seg_n, char * const buff, unsigned int * num_split, const char *  split_type);
//字符串分割函数,返回字整型

void get_train_flavor(char ** const buff, char ** const t_flavor, int data_num);
//获取train文件的虚拟机类型
void get_train_flavornum(char ** const t_flavor, unsigned int * t_flavor_num, int data_num);
//获取train文件的每种虚拟机数量
void get_train_timestamp(char ** const buff, char ** const t_time_stamp,int data_num);
//获取train文件的时间戳
void get_flavornum_day(int data_num);
//获取train文件某一天某种虚拟机数量

void get_CPUcore_num(char ** const buff, unsigned int * core_num);
//获取物理服务器CPU内核数
void get_RAM(char ** const buff, unsigned int * RAM);
//获取物理服务器内存大小
void get_DISK(char ** const buff, unsigned int * DISK);
//获取物理服务器硬盘容量

void get_input_flavorN(char ** const buff, unsigned int *i_flavor_N);
//获取虚拟机总数
unsigned int get_opt_type(char ** const buff);
//获取优化选项;  True:CPU  FALSE:MEM
void get_input_flavor(char ** const buff,char ** const i_flavor);
//获取input文件的虚拟机类型
void get_input_flavorcore(char ** const buff,unsigned int * const i_flavor_num);
//获取input文件的虚拟机内核数量
void get_input_fRAM(char ** const buff,unsigned int * const i_flavor_ram);
//获取input文件的虚拟机内存大小
void predict_day(char ** const buff, char ** const i_flavor);
//获取input文件的时间戳

void get_function(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num);//信息获取总调用函数

#endif
