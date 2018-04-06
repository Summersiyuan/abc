/* 需要解决的问题：
 * 1.接口
 * 2.计算规定类型的虚拟机
 * 3.要考虑箱子有不同的规格
 * 4.输出函数还没有完成
 * 5.物理服务器中内存不能为单数(实际上不存在)
 *
 */

#include "function.h"
#include "predict.h"
#include "get_info.h"
#include "average.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/timeb.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <lib_io.h>

int predict_server_all=0;//预测虚拟机（CPU或MEM）总容量
int number_server = 0;//需要物理服务器的数目
int output_i[F][N];//[服务器]、[虚拟机型号]、虚拟机个数
//#define N 15 //需要测试的虚拟机个数

void put(unsigned int core_num,unsigned int RAM, int  flavor[][3],unsigned int opti_type) 
{
	printf("Chengcong is excellent!\n\n");
	int i = 0, j = 0;
	int total_CPU = 0, total_MEM = 0;//CPU和MEM的总量
	//int flavor[N][3];//分别代表不同虚拟机的个数、CPU和MEM
	int core_num_i= core_num;
	int RAM_i= RAM;
	int server[2] = {core_num_i , RAM_i};//物理服务器规格
	int minimal_server = 0;//需要物理服务器的最小值
	int capacity_server[2] = { 0, 0};//物理服务器的容量
	int remain = 0;//剩余虚拟机总数
	int actual_server_all;//实际物理服务器（CPU或MEM）总容量
	int check_CPU;//优化CPU和MEM的判定条件之一
	int check_1;
	double score;//得分

	for (i = 0; i < F; i++)//空数组
	{
		for (j = 0; j < N; j++)
			output_i[i][j] = 0;
	}

	//记录输入的虚拟机个数，cpu核数，内存（GB）
	unsigned int exchange[N] = {0};//测试数据为2015年1月和2月数据
	for (i = 0; i < N; i++)
	{
		remain += flavor[i][0];//更新剩余虚拟机总数
		exchange[i] = flavor[i][0];//计数
		//flavor[i][1] = my_pow(2, (int)(i / 3));//计算CPU
		//计算MEM
		//if (i % 3 == 0)
		//	flavor[i][2] = flavor[i][1];
		//else if (i % 3 == 1)
		//	flavor[i][2] = flavor[i][1] * 2;
		//else
		//	flavor[i][2] = flavor[i][1] * 4;

		total_CPU += flavor[i][0] * flavor[i][1];//计算CPU和MEM总数
		total_MEM += flavor[i][0] * flavor[i][2];

	}

	if (my_ceil(total_CPU / server[0]) >= my_ceil(total_MEM / server[1]))
	{
		minimal_server = my_ceil(total_CPU / server[0]);
		printf("At least %d physical servers are needed\n", minimal_server);
	}
	else
	{
		minimal_server = my_ceil(total_MEM / server[1]);
		printf("At least %d physical servers are needed\n", minimal_server);
	}

	printf("There are %d virtual servers in total\n\n", remain);

	if (opti_type==CPU || opti_type==MEM)//优化CPU，放置虚拟机
	{
		while (remain > 0) //有虚拟机剩余时执行
		{
			i = N - 1;
			while(i >= 0)
			{
				check_CPU = 0;

				if (i == 0 && flavor[0][0] == 0)
					check_1 = 1;
				else
					check_1 = 0;
				
				if ((flavor[i][0] > 0) || (i == 0))
				{
					if ((capacity_server[0] + flavor[i][1] < server[0]) && (capacity_server[1] + \
                                                flavor[i][2] == server[1]) && (check_1 == 0))
					{

						if ((flavor[i][1] / flavor[i][2]) == 1)//如果虚拟机的
							check_CPU = 0;
						else
							check_CPU = 1;
					}
	
					if ((check_CPU == 0) && (capacity_server[0] + flavor[i][1] <= server[0]) && \
						(capacity_server[1] + flavor[i][2] <= server[1]) && (check_1 == 0))//CPU和MEM都不会超
					{
						capacity_server[0] += flavor[i][1];//加上CPU
						capacity_server[1] += flavor[i][2];//加上MEM
						
						output_i[number_server][i] += 1;//输出文件中虚拟机个数累加
						flavor[i][0] -= 1;//第i个虚拟机的数量减少
						remain -= 1;//虚拟机总数减少
					}
					else
					{
						i -= 1; //printf("%d\n", i);
						if (i < 0) 
						{
							if (remain > 0)
								number_server += 1;//放不下了，新开一台物理服务器

							capacity_server[0] = 0;//新的物理服务器CPU和MEM清0
							capacity_server[1] = 0;

							//i = N - 1;//启动一台新的物理服务器的时候又要从大的虚拟服务器开始装起
						}
					}	
				}
				else
				{
					i -= 1;
				}
			}
		}
	}
	//输出函数
	printf("%d\n", number_server + 1);
	for (i = 0; i <= number_server; i++)
	{
		printf("%d", i + 1);
		for (j = 0; j < N; j++)
		{
			if (output_i[i][j] != 0) 
			{
				printf(" flavor%d %d", j + 1, output_i[i][j]);
			}
		}
		printf("\n");
	}

	//计算得分
	predict_server_all = 0;
	actual_server_all = 0;

	if (opti_type==CPU) 
	{
		for (i = 0; i < N; i++)
		{
			actual_server_all += exchange[i] * flavor[i][1];
		}
		predict_server_all = (number_server + 1) * server[0];

		score = (double)((double)(actual_server_all) / (double)(predict_server_all));

		printf("%d\n", predict_server_all);
		printf("%d\n", actual_server_all);

		printf("\nThis optimization is %.2lf points for CPU\n\n", score * 100);
	}
	else if (opti_type==MEM)
	{
		for (i = 0; i < N; i++)
		{
			actual_server_all += exchange[i] * flavor[i][2];
		}
		predict_server_all = (number_server + 1) * server[1];

		score = (double)((double)(actual_server_all) / (double)(predict_server_all));

		printf("%d\n", predict_server_all);
		printf("%d\n", actual_server_all);

		printf("\nThis optimization is %.2lf points for MEM\n\n", score * 100);
	}
	printf("Chengcong is excellent!\n\n");
	
	
}

int my_pow(int x, int y)//求x的y次幂
{
	int m = 1;
	if (y == 0)
		m = 1;
	else 
		for (int i = 0; i < y; i++)
			m = m * 2;
	return m;
}

double my_ceil(double x)//向上取整函数
{
	if (x - int(x) == 0)
		return (float)((int)x);
	else
		return (float)((int)x) + 1;
}



