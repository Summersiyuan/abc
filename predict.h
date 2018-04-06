#ifndef __ROUTE_H__
#define __ROUTE_H__

#include "lib_io.h"


void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename);
//spec为允许解析的最大行数,返回每行信息

//extern char **read_file_line(char ** const buff, const unsigned int spec, const char * const filename);
//读取文件每行信息 输入buff缓存区 最大数据条数 文件路径
//输出每行信息


#endif
