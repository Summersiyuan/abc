#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "predict_gap.h"
#include "least_square_method.h"
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
#include "math.h" 

//#define LEAST_KALMAN //最小二乘法加滤波, 屏蔽不加滤波

#ifdef _DEBUG
#define PRINT   printf
#else
#define PRINT(...)
#endif


#define ParaBuffer(Buffer,Row,Col) (*(Buffer + (Row) * (SizeSrc + 1) + (Col)))  
 /***********************************************************************************  
 ***********************************************************************************/  
// static int GetXY(const char* FileName, double* X, double* Y, int* Amount)  
// {  
//     FILE* File = fopen(FileName, "r");  
//     if (!File)  
//        return -1;  
//     for (*Amount = 0; !feof(File); X++, Y++, (*Amount)++)  
//         if (2 != fscanf(File, (const char*)"%lf %lf", X, Y))  
//            break;  
//     fclose(File);  
//     return 0;  
// }  
 /***********************************************************************************  
 ***********************************************************************************/  
// static int PrintPara(double* Para, int SizeSrc)  
// {  
//     int i, j;  
//     for (i = 0; i < SizeSrc; i++)  
//     {  
//         for (j = 0; j <= SizeSrc; j++)  
//         printf("%10.6lf ", ParaBuffer(Para, i, j));  
//         printf("\r\n");  
//     }  
//     printf("\r\n");  
//     return 0;  
// }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParalimitRow(double* Para, int SizeSrc, int Row)  
 {  
     int i;  
     double Max, Min, Temp;  
     for (Max = abs(ParaBuffer(Para, Row, 0)), Min = Max, i = SizeSrc; i; i--)  
     {  
         Temp = abs(ParaBuffer(Para, Row, i));  
         if (Max < Temp)  
             Max = Temp;  
         if (Min > Temp)  
             Min = Temp;  
     }  
     Max = (Max + Min) * 0.000005;  
     for (i = SizeSrc; i >= 0; i--)  
         ParaBuffer(Para, Row, i) /= Max;  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int Paralimit(double* Para, int SizeSrc)  
 {  
     int i;  
     for (i = 0; i < SizeSrc; i++)  
          if (ParalimitRow(Para, SizeSrc, i))  
             return -1;  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParaPreDealA(double* Para, int SizeSrc, int Size)  
 {  
     int i, j;  
     for (Size -= 1, i = 0; i < Size; i++)  
     {  
         for (j = 0; j < Size; j++)  
         ParaBuffer(Para, i, j) = ParaBuffer(Para, i, j) * ParaBuffer(Para, Size, Size) - ParaBuffer(Para, Size, j) * ParaBuffer(Para, i, Size);  
         ParaBuffer(Para, i, SizeSrc) = ParaBuffer(Para, i, SizeSrc) * ParaBuffer(Para, Size, Size) - ParaBuffer(Para, Size, SizeSrc) * ParaBuffer(Para, i, Size);  
         ParaBuffer(Para, i, Size) = 0;  
         ParalimitRow(Para, SizeSrc, i);  
     }  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParaDealA(double* Para, int SizeSrc)  
 {  
     int i;  
     for (i = SizeSrc; i; i--)  
         if (ParaPreDealA(Para, SizeSrc, i))  
             return -1;  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParaPreDealB(double* Para, int SizeSrc, int OffSet)  
 {  
     int i, j;  
     for (i = OffSet + 1; i < SizeSrc; i++)  
     {  
         for (j = OffSet + 1; j <= i; j++)  
         ParaBuffer(Para, i, j) *= ParaBuffer(Para, OffSet, OffSet);  
         ParaBuffer(Para, i, SizeSrc) = ParaBuffer(Para, i, SizeSrc) * ParaBuffer(Para, OffSet, OffSet) - ParaBuffer(Para, i, OffSet) * ParaBuffer(Para, OffSet, SizeSrc);  
         ParaBuffer(Para, i, OffSet) = 0;  
         ParalimitRow(Para, SizeSrc, i);  
     }  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParaDealB(double* Para, int SizeSrc)  
 {  
     int i;  
     for (i = 0; i < SizeSrc; i++)  
             if (ParaPreDealB(Para, SizeSrc, i))  
                     return -1;  
     for (i = 0; i < SizeSrc; i++)  
     {  
         if (ParaBuffer(Para, i, i))  
         {  
             ParaBuffer(Para, i, SizeSrc) /= ParaBuffer(Para, i, i);  
             ParaBuffer(Para, i, i) = 1.0;  
         }  
     }  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int ParaDeal(double* Para, int SizeSrc)  
 {  
     //PrintPara(Para, SizeSrc);  
     Paralimit(Para, SizeSrc);  
     //PrintPara(Para, SizeSrc);  
     if (ParaDealA(Para, SizeSrc))  
             return -1;  
     //PrintPara(Para, SizeSrc);  
     if (ParaDealB(Para, SizeSrc))  
             return -1;  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 static int GetParaBuffer(double* Para, const double* X, const double* Y, int Amount, int SizeSrc)  
 {  
     int i, j;  
     for (i = 0; i < SizeSrc; i++)  
             for (ParaBuffer(Para, 0, i) = 0, j = 0; j < Amount; j++)  
                     ParaBuffer(Para, 0, i) += pow(*(X + j), 2 * (SizeSrc - 1) - i);  
     for (i = 1; i < SizeSrc; i++)  
             for (ParaBuffer(Para, i, SizeSrc - 1) = 0, j = 0; j < Amount; j++)  
                     ParaBuffer(Para, i, SizeSrc - 1) += pow(*(X + j), SizeSrc - 1 - i);  
     for (i = 0; i < SizeSrc; i++)  
             for (ParaBuffer(Para, i, SizeSrc) = 0, j = 0; j < Amount; j++)  
                     ParaBuffer(Para, i, SizeSrc) += (*(Y + j)) * pow(*(X + j), SizeSrc - 1 - i);  
     for (i = 1; i < SizeSrc; i++)  
             for (j = 0; j < SizeSrc - 1; j++)  
                     ParaBuffer(Para, i, j) = ParaBuffer(Para, i - 1, j + 1);  
     return 0;  
 }  
 //***********************************************************************************  
 //***********************************************************************************  
 int Cal(const double* BufferX, const double* BufferY, int Amount, int SizeSrc, double* ParaResK)  
 {  
     double* ParaK = (double*)malloc(SizeSrc * (SizeSrc + 1) * sizeof(double));  
     GetParaBuffer(ParaK, BufferX, BufferY, Amount, SizeSrc);  
     ParaDeal(ParaK, SizeSrc);  
     for (Amount = 0; Amount < SizeSrc; Amount++, ParaResK++)  
             *ParaResK = ParaBuffer(ParaK, Amount, SizeSrc);  
     free(ParaK);  
     return 0;  
 }  
 /***********************************************************************************  
 ***********************************************************************************/  
 void least_square(int flavor_type)  
 {  
     int Amount=t_gap+1;  
     clock_t StartTime, FinishTime; //  声明两个时间变量  
     double DiffTime;  
     StartTime = clock();       //  开始计时  

     double BufferX[MAX_TRAIN_DAYS],  ParaK[3]; //  5次拟合, 一共6个系数(包含常数项)  


#ifdef LEAST_KALMAN //最小二乘法加滤波
     double BufferY_MUL[MAX_TRAIN_DAYS];
	 double flavornum_day_f[MAX_TRAIN_DAYS];
     for (int i = 1; i <  MAX_TRAIN_DAYS ; i++)
	 {
	 	BufferX[i]=BufferX[i-1]+1;
	 	//printf("BufferX[%d]: %f\n", i-1, BufferX[i-1]); 
	 }
	 
	 int flavornumday_temp[MAX_TRAIN_DAYS];
     for (int i = 0; i <  MAX_TRAIN_DAYS ; i++)
	 {
	 	flavornumday_temp[i]=flavornum_day[i][predict_flavor[flavor_type]-1];
	 	//printf("BufferX[%d]: %f\n", i-1, BufferX[i-1]); 
	 }

	 kalman(flavornumday_temp, t_gap, flavornum_day_f, 0.0001, 0.0001, 1);//Kalman滤波
	 memcpy( BufferY_MUL , flavornum_day_f, sizeof(flavornum_day_f));
	 
 
     // 读入要拟合的数据  
     //if (GetXY((const char*)"test.txt", (double*)BufferX, (double*)BufferY, &Amount))  
       //   return 0;  
     double  BufferY[MAX_TRAIN_DAYS];
     printf("flavor_type=%d",predict_flavor[flavor_type]);
     for (int i = 0; i <  Amount ; i++)
	 {
	 	BufferY[i]=BufferY_MUL[i];
	 	//if(BufferY[i]>10)
	 		//BufferY[i]=10;
	 	//printf("BufferY[%d]: %f\n", i, BufferY[i]);
	 }

#else	//不加滤波
     int BufferY_MUL[MAX_TRAIN_DAYS][N]; 
     for (int i = 1; i <  MAX_TRAIN_DAYS ; i++)
	 {
	 	BufferX[i]=BufferX[i-1]+1;
	 	//printf("BufferX[%d]: %f\n", i-1, BufferX[i-1]); 
	 }
	 memcpy( BufferY_MUL , flavornum_day, sizeof(flavornum_day));
     // 读入要拟合的数据  
     //if (GetXY((const char*)"test.txt", (double*)BufferX, (double*)BufferY, &Amount))  
       //   return 0;  
     double  BufferY[MAX_TRAIN_DAYS];
     //printf("flavor_type=%d",predict_flavor[flavor_type]);
     for (int i = 0; i <  Amount ; i++)
	 {
	 	BufferY[i]=(double)BufferY_MUL[i][predict_flavor[flavor_type]-1];
	 	if(BufferY[i]>15)    //截断异常过大数据
	 		BufferY[i]=15;
	 	//printf("BufferY[%d]: %f\n", i, BufferY[i]);
	 }
#endif
     
     printf("Amount: %d\n", Amount);  
     Cal((const double*)BufferX, (const double*)BufferY, Amount, sizeof(ParaK) / sizeof(double), (double*)ParaK);  //拟合
     //printf("拟合系数为:\n");  
     //printf("按升序排列\n");  
     //for (unsigned i = 0; i <  sizeof(ParaK) / sizeof(double); i++)  
             //printf("ParaK[%d] = %lf\r\n", i, ParaK[i]);  
             
     int gap_temp=get_time_gap(t_time_stamp[0], predict_daystamp[0] ); //计算预测开始时间与train文件第一天的间隔
     
     //printf("%d\n",gap_temp);
     for (int i = gap_temp; i < gap_temp+i_gap ; i++)//最小二乘法得到曲线方程求值
	 {
     	flavor[predict_flavor[flavor_type]-1][0]+= (int) ParaK[2]+ParaK[1]*BufferX[i]+\
     	ParaK[0]*pow(BufferX[i],2);

     }
     
     
     double flavor_temp[N]={0}, average_flavor=0.0;
     if(flavor[predict_flavor[flavor_type]-1][0]<0||(predict_flavor[flavor_type]-1)==12||(predict_flavor[flavor_type]-1)==14)  //处理拟合得到的赋值,当拟合得到的值为赋值时采用平均值
     {
     	
		for(unsigned int i=0;i<N;i++)
		{
			for(int j=0;j<t_gap+1;j++)
			{
				//kalman(flavornum_day, i, flavornum_day_f, 0.001, 0.1, 1);//Kalman滤波
		    	flavor_temp[i]+=(double)flavornum_day[j][i];
		   	}
		}
     	//flavor[predict_flavor[flavor_type]-1][0]=0;
     	average_flavor=(double)flavor_temp[predict_flavor[flavor_type]-1]/(t_gap+1);//求train文件内对应虚拟机的每天平均使用数量
		flavor[predict_flavor[flavor_type]-1][0]=average_flavor*(i_gap+1);
     }
     //if(predict_flavor[flavor_type]==14||predict_flavor[flavor_type]==11||predict_flavor[flavor_type]==16)
		//flavor[predict_flavor[flavor_type]-1][0]=0;
     
     FinishTime = clock();  //  结束计时  
     DiffTime = FinishTime - StartTime; //拟合时间  
     printf("拟合时间为: %lf\n", DiffTime);  
     //return 0;  
 }
 
 
void least_square_mul(unsigned int i_flavor_N)
{
	for(unsigned int i=0; i<i_flavor_N;i++) //拟合需要预测的虚拟机
	{
 		least_square(i); //对每一个虚拟机数量拟合
 	}
 	
/*************输出数据kalman滤波**********************/
 /*	int flavor_int[N];
    double flavor_f2[N];
     for (int i = 0; i <  N ; i++)
	 {
	 	flavor_int[i]=flavor[i][0];
	 	//printf("BufferX[%d]: %f\n", i-1, BufferX[i-1]); 
	 }

     kalman(flavor_int, N, flavor_f2, 0.0001, 0.0001, 1);//Kalman滤波
     for (unsigned int i = 0; i <  i_flavorN ; i++)
	 {
	 	flavor[predict_flavor[i]-1][0]=flavor_f2[predict_flavor[i]-1];
	 	//printf("BufferX[%d]: %f\n", i-1, BufferX[i-1]); 
	 }
*/
}
