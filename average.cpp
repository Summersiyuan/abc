#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "predict_gap.h"
#include "kalman.h"
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



#ifdef _DEBUG
#define PRINT   printf
#else
#define PRINT(...)
#endif

//#define KALMAN


unsigned int predict_flavor[N];//需要预测的虚拟机型号

void average(unsigned int * t_flavor_num, char ** const i_flavor, 
			 int  class_flavor[][3], unsigned int i_flavor_N,int data_num)
{
	unsigned int  flavor_num=0;
	double average_flavor=0.0, flavor_temp[N]={0};
#ifdef KALMAN 	
	double flavornum_day_f[MAX_TRAIN_DAYS];
    for(unsigned int i=0;i<N;i++)
    {
    	for(int j=0;j<t_gap+1;j++)
    	{
			kalman(flavornum_day, i, flavornum_day_f, 0.001, 0.1, 1);//Kalman滤波
        	flavor_temp[i]+=(float)flavornum_day_f[j];
       	}
    }
#else
	    for(unsigned int i=0;i<N;i++)
    {
    	for(int j=0;j<t_gap+1;j++)
    	{
			//kalman(flavornum_day, i, flavornum_day_f, 0.001, 0.1, 1);//Kalman滤波
        	flavor_temp[i]+=(float)flavornum_day[j][i];
       	}
    }
#endif
		for(unsigned int i=0; i<i_flavor_N;i++)
		{
			if(strlen(i_flavor[i])==7)
			{
				flavor_num=i_flavor[i][6]-'0';//得到input文件的虚拟机编号
			}
			else if(strlen(i_flavor[i])==8)
			{
				flavor_num=10*(i_flavor[i][6]-'0')+i_flavor[i][7]-'0';//得到input文件的虚拟机编号
			}
			//printf("%d\n",flavor_num);
			//predict_flavor[i]=flavor_num;
			average_flavor=(double)flavor_temp[flavor_num-1]/(t_gap+1);
			//printf("%d\n",t_flavor_num[0]);
			//printf("%d\n",average_flavor);

			class_flavor[(flavor_num-1)][0]=average_flavor*(i_gap+1);//求train文件内对应虚拟机的每天平均使用数量
		}

}

