#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "vector"
#include "assert.h"
#include "string.h"
#include "Correlation.h"
#include "get_info.h"

//自相关计算函数; data:输入数据 m:时间差长度 N:数据长度
int AutoCorrelation(int* data, int m, int N2)  
{  
     int r=0;  

     for(int i = m; i <N2;i++ )  
     {  
          r +=data[i] * data[i-m];  

     }  

     return  r ;  
}  

void  myAutoCorrelation(int data[][N], int* Corr)  
{  
	 int data1[t_gap+1];
	 for(int k=0; k<t_gap+1; k++)
	 	{
	 		data1[k]=data[k][0];
	 		//printf("%d\n", data1[k]);
	 	}
	 int N1=t_gap+1;
     for(int m=0; m <N1-1; m++ )  
     {  
          Corr[m]=AutoCorrelation(data1, m, N1); 

     }  

     //return  Corr ;  
}  

