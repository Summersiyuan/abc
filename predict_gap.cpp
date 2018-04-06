#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "predict_gap.h"
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


/* 返回绝对值 */
int abs(int a,int b)
{
 if(a>=b)
  return (a-b);
 else
  return (b-a);
}
/* 判断是否为闰年:是,返回 1 ; 不是, 返回 0 . */
int IsLeap(int year)
{
 if(((year%4==0)&&(year%100!=0))||year%400==0)
  return 1;
 else
  return 0;
}
/* 判断某个日期从年初(y年1月1日)到该天(y年m月d日)的天数 */
int Days(int y,int m,int d)
{
 int days[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
 int i;
 int sum = 0 ;  /* 计算天数 */
 if(IsLeap(y)) /* 如果为闰年,2月有 29 天 */
  days[2] = 29 ;
 for(i=0;i<m;i++)
  sum = sum +days[i] ;
 sum = sum + d - 1 ;
  return sum;
}
/*
 判断两个日期之间的天数.
 日期一:年 y1,月 m1,日 d1;
 日期一:年 y2,月 m2,日 d2;
*/
int DiffDays(int y1,int m1,int d1,int y2,int m2,int d2)
{
 int s1,s2; /* 计算两个日期从年初到该日期的天数 */
 int count; /* 计算两个年份之间的差值 */
 int sum=0; /*                        */
 int t,t1,t2;
 if(y1==y2)
 {
  s1 = Days(y1,m1,d1);
  s2 = Days(y2,m2,d2);
  return abs(s1,s2);  
 }
 else if(y1>y2)
 {
  count = y1 - y2 ;
  if(count == 1)
  {
   t1 = Days(y1,m1,d1);
   t2 = Days(y2,12,31) - Days(y2,m2,d2);
   return (t1+t2+count);
  }
  else
  {   
   for(t = y2+1;t<y1;t++)
    sum = sum + Days(t,12,31);
   //cout<<sum<<endl;
   t2 = Days(y2,12,31) - Days(y2,m2,d2);
   t1 = Days(y1,m1,d1);
   return (sum+t1+t2+count);
  }
 }
 else
 {
  count = y2 - y1 ;
  if(count == 1)
  {
   t2 = Days(y2,m2,d2);
   t1 = Days(y1,12,31) - Days(y1,m1,d1);
   return (t1+t2+count);
  }
  else
  {   
   for(t = y1+1;t<y2;t++)
    sum = sum + Days(t,12,31);
   t2 = Days(y2,m2,d2);
   t1 = Days(y1,12,31) - Days(y1,m1,d1);
   return (sum+t1+t2+count);
  }
 }
  
} 

int get_time_gap(char * const timestamp0, char * const timestamp1 )
{
	unsigned int year1, year0, month1, month0, day1 ,day0;
	const char * splittype= "-";
	int time_gap;
	year1=int_split(1, timestamp1, &year1, splittype);
	year0=int_split(1, timestamp0, &year0, splittype);
	month1=int_split(2, timestamp1, &month1, splittype);
	month0=int_split(2, timestamp0, &month0, splittype);
	day1=int_split(3, timestamp1, &day1, splittype);
	day0=int_split(3, timestamp0, &day0, splittype);
	int y1, y0, m1, m0, d1 ,d0;
	y1=(int) year1;
	y0=(int) year0;
	m1=(int) month1;
	m0=(int) month0;
	d1=(int) day1;
	d0=(int) day0;
	time_gap=DiffDays(y1,m1,d1,y0,m0,d0);//得到预测时间跨度
	return time_gap;
}
