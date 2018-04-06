#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "predict_gap.h"
#include "Correlation.h"
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

char *t_flavor[MAX_DATA_NUM]; //train文件的虚拟机类型数组
char *t_time_stamp[MAX_DATA_NUM];//train文件每个虚拟机对应的请求时间
unsigned int t_flavor_num[N]; //train文件里面每一类型虚拟机总量
int t_gap, i_gap; //train文件的时间间隔 input文件的预测时间间隔
unsigned int core_num; //物理服务器内核数量
unsigned int RAM; //物理服务器内存
unsigned int DISK; //物理服务器硬盘
unsigned int i_flavorN; //预测虚拟机种类数
unsigned int type_opti; //优化选项
int flavornum_day[MAX_TRAIN_DAYS][N]={0}; //[train数据某一天][虚拟机型号] 某一天某种虚拟机使用数量
int flavornum_day_cheng[MAX_TRAIN_DAYS][N]={0}; //[train数据某一天][0-i_flavorN] 某一天某种虚拟机使用数量
char *i_flavor[MAX_DATA_NUM]; 
unsigned int i_flavor_core[MAX_INFO_NUM]; 
unsigned int i_flavor_RAM[MAX_INFO_NUM];
char *predict_daystamp[2];//预测时间戳信息
//ÄãÒªÍê³ÉµÄ¹ŠÄÜ×ÜÈë¿Ú




char* str_split(unsigned char seg_n, char * const buff, char * str_split, const char *  split_type)
{
    unsigned int cnt=0;
    char str[MAX_LINE_LEN];
    char *p;
    strcpy(str, buff);
    p = strtok(str,split_type); 
    while(p!=NULL) 
    {   
	    cnt++;
	    if(cnt==seg_n)
	    {	
		//t_flavor[i] = (char *)malloc(MAX_LINE_LEN + 2);
	        strcpy(str_split, p);
		//printf ("%s\n",t_flavor[i]);
	    }
            p = strtok(NULL,split_type);
    }
    return str_split;
}

unsigned int int_split(unsigned char seg_n, char * const buff, unsigned int * num_split, const char *  split_type)
{
    unsigned int cnt=0;
    char str[MAX_LINE_LEN];
    char *p;
    strcpy(str, buff);
    p = strtok(str,split_type); 
    while(p!=NULL) 
    {   
	    cnt++;
	    if(cnt==seg_n)
	    {	
		//t_flavor[i] = (char *)malloc(MAX_LINE_LEN + 2);
	        //strcpy(str_split, p);
		*num_split=atoi(p);
		//printf ("%s\n",t_flavor[i]);
	    }
            p = strtok(NULL,split_type);
    }
    return *num_split;
}

void get_train_flavor(char ** const buff, char ** const t_flavor, int data_num)
{
    const char * split = "	";    
    unsigned char seg_n = 2;
    //char *t_flavor[MAX_DATA_NUM]; 
    for(int i=0;i<data_num;i++)
    {
		t_flavor[i] = (char *)malloc(MAX_LINE_LEN + 2);	
		t_flavor[i] = str_split(seg_n, buff[i], t_flavor[i], split);
    }
    //return 0;  
}

void get_train_flavornum(char ** const t_flavor, unsigned int * t_flavor_num, int data_num)
{
	unsigned int j=0,k=0;
    t_flavor_num[N]={0};
    for(int i=0;i<data_num;i++)
    {
		if(strlen(t_flavor[i])==7)
		{
			j=t_flavor[i][6]-'0';
			//printf ("%d\n",j);
			switch(j)
			{ 
				case 1: t_flavor_num[0]++; break;
				case 2: t_flavor_num[1]++; break;
				case 3: t_flavor_num[2]++; break;
				case 4: t_flavor_num[3]++; break;
				case 5: t_flavor_num[4]++; break;
				case 6: t_flavor_num[5]++; break;
				case 7: t_flavor_num[6]++; break;
				case 8: t_flavor_num[7]++; break;
				case 9: t_flavor_num[8]++; break;
				//default:  printf("虚拟机种类不止15个\n");
			}
		}
		else if(strlen(t_flavor[i])==8)
		{
			k=t_flavor[i][6]-'0';
			j=t_flavor[i][7]-'0';
			//printf ("%d\n",k);
			//printf ("%d\n",j);
			//printf ("%s\n",t_flavor[i]);
			if(k==1)
			{
				//printf ("%d\n",k);
				switch((j+10))
				{ 
					 case 10: t_flavor_num[9]++;  break;
					 case 11: t_flavor_num[10]++; break;
					 case 12: t_flavor_num[11]++; break;
					 case 13: t_flavor_num[12]++; break;
					 case 14: t_flavor_num[13]++; break;
					 case 15: t_flavor_num[14]++; break;
					 case 16: t_flavor_num[15]++; break;
					 case 17: t_flavor_num[16]++; break;
					 case 18: t_flavor_num[17]++; break;
					 case 19: t_flavor_num[18]++; break;
					 //default:  printf("虚拟机种类不止15个\n");
				}
			}
			else if (k==2)
			{
				//printf ("%d\n",k);
				//printf ("%d\n",j);
				//printf ("%s\n",t_flavor[i]);
				switch((j+20))
				{ 
					 case 20: t_flavor_num[19]++; break;
					 case 21: t_flavor_num[20]++; break;
					 case 22: t_flavor_num[21]++; break;
					 case 23: t_flavor_num[22]++; break;
					 case 24: t_flavor_num[23]++; break;
					 case 25: t_flavor_num[24]++; break;
					 default:  printf("虚拟机种类不止%d个\n",N);
				}
			}
		}
	}
	//printf("%s\n",t_time_stamp[3]);
    int r = 0;
    for(unsigned int i=0;i<N;i++)
        r+=t_flavor_num[i];
    if(r==data_num)
	printf("读取train文件正确\n");
    else
	printf("读取train文件错误,错误数总数为%d\n",r);
}

void get_train_timestamp(char ** const buff,char ** const t_time_stamp, int data_num)
{
    const char * split = "	";    
    unsigned char seg_n = 3;  
    for(int i=0;i<data_num;i++)
    {
	t_time_stamp[i] = (char *)malloc(MAX_LINE_LEN + 2);	
	
	t_time_stamp[i] = str_split(seg_n, buff[i], t_time_stamp[i], split);
    }
    //return 0;  
}


void get_CPUcore_num(char ** const buff,unsigned int * core_num)
{
    const char * split = " ";   
    unsigned char seg_n = 1;  
    *core_num = int_split(seg_n, buff[0], core_num, split);
    //return core_num;  
}

void get_RAM(char ** const buff, unsigned int * RAM)
{
    const char * split = " ";   
    unsigned char seg_n = 2;  
    *RAM = int_split(seg_n, buff[0], RAM, split);
    //return core_num;  
}

void get_DISK(char ** const buff,unsigned int * DISK)
{
    const char * split = " ";   
    unsigned char seg_n = 3;  
    *DISK = int_split(seg_n, buff[0], DISK, split);
    //return core_num;    
}

void get_input_flavorN(char ** const buff, unsigned int *i_flavor_N)
{
    const char * split = " ";   
    unsigned char seg_n = 1;  
    *i_flavor_N = int_split(seg_n, buff[2], i_flavor_N, split);
    //return core_num;   
}

unsigned int get_opt_type(char ** const buff)
{  
    char type[MAX_LINE_LEN]="";
    unsigned int type_flag=0;   
    for(unsigned int i=0; i<8+i_flavorN; i++)
    {
	if(strlen(buff[i])==2)
	{
	    strcpy(type, buff[i+1]);
	    //type_flag=TRUE;
	    if(strncmp(type,"CPU",3)==0)
		{
		    type_flag=CPU;
		    //return TRUE;
		    break;
		}
	    else if(strncmp(type,"MEM",3)==0)
		{
		    type_flag=MEM;
		    //return FALSE;
		    break;
		}
	 }
    } 
    return type_flag;
}

void get_input_flavor(char ** const buff,char ** const i_flavor)
{
    const char * split = " ";    
    unsigned char seg_n = 1;
    unsigned int i=2;
    while(strlen(buff[i])!=2)
    {
		i_flavor[i-2] = (char *)malloc(MAX_LINE_LEN + 2);	
		i_flavor[i-2] = str_split(seg_n, buff[i+1], i_flavor[i-2], split);
		i++;
    }
    unsigned int  flavor_num=0;
    for(unsigned int i=0; i<i_flavorN;i++)
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
		predict_flavor[i]=flavor_num;
	 }
    //return 0;  
}


void get_input_flavorcore(char ** const buff,unsigned int * const i_flavor_num)
{
    const char * split = " ";    
    unsigned char seg_n = 2;
    unsigned int i=2;
    while(strlen(buff[i])!=2)
    {
	//i_flavor_num[i-2] = (unsigned int *)malloc(MAX_LINE_LEN + 2);	
	i_flavor_num[i-2] = int_split(seg_n, buff[i+1], &i_flavor_num[i-2], split);
	i++;
    }
    //return 0;  
}

void get_input_fRAM(char ** const buff,unsigned int * const i_flavor_ram)
{
    const char * split = " ";    
    unsigned char seg_n = 3;
    unsigned int i=2;
    while(strlen(buff[i])!=2)
    {
	//i_flavor_num[i-2] = (unsigned int *)malloc(MAX_LINE_LEN + 2);	
	i_flavor_ram[i-2] = int_split(seg_n, buff[i+1], &i_flavor_ram[i-2], split);
	i++;
    }
    //return 0;  
}

void predict_day(char ** const buff,char ** const i_time_stamp)
{
    const char * split = " ";     
    unsigned int i = 0,blank_num=0,start_day_flag=0;  
    unsigned char seg_n = 1;   
    while(blank_num!=3)
    {
	if(strlen(buff[i])==2)	
	{
	    blank_num++;
	    if(blank_num==3)
	        {
	            start_day_flag=i+1; 
	        }
	}
	i++;
    }
    for(unsigned int i=0; i<2; i++)
    {
	i_time_stamp[i] = ( char *)malloc(MAX_LINE_LEN + 2);	
	i_time_stamp[i] = str_split(seg_n, buff[start_day_flag], i_time_stamp[i], split);
	start_day_flag++;
    }
    //return 0;  
}

void get_flavornum_day(int data_num)
{
	unsigned int j=0,k=0;
    //t_flavor_num[N]={0};
    for(int i=0;i<data_num;i++)
    {
		int t_gap_num=get_time_gap(t_time_stamp[0], t_time_stamp[i] );
		if(strlen(t_flavor[i])==7)
		{
			j=t_flavor[i][6]-'0';
			//printf ("%d\n",j);
			switch(j)
			{ 
				case 1: flavornum_day[t_gap_num][j-1]++; break;
				case 2: flavornum_day[t_gap_num][j-1]++; break;
				case 3: flavornum_day[t_gap_num][j-1]++; break;
				case 4: flavornum_day[t_gap_num][j-1]++; break;
				case 5: flavornum_day[t_gap_num][j-1]++; break;
				case 6: flavornum_day[t_gap_num][j-1]++; break;
				case 7: flavornum_day[t_gap_num][j-1]++; break;
				case 8: flavornum_day[t_gap_num][j-1]++; break;
				case 9: flavornum_day[t_gap_num][j-1]++; break;
				//default:  printf("虚拟机种类不止15个\n");
			}
		}
		else if(strlen(t_flavor[i])==8)
		{
			k=t_flavor[i][6]-'0';
			j=t_flavor[i][7]-'0';
			//printf ("%d\n",k);
			//printf ("%d\n",j);
			//printf ("%s\n",t_flavor[i]);
			if(k==1)
			{
				//printf ("%d\n",k);
				switch((j+10))
				{ 
					 case 10: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 11: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 12: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 13: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 14: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 15: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 16: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 17: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 18: flavornum_day[t_gap_num][j+10-1]++; break;
					 case 19: flavornum_day[t_gap_num][j+10-1]++; break;
					 //default:  printf("虚拟机种类不止15个\n");
				}
			}
			else if (k==2)
			{
				//printf ("%d\n",k);
				//printf ("%d\n",j);
				//printf ("%s\n",t_flavor[i]);
				switch((j+20))
				{ 
					 case 20: flavornum_day[t_gap_num][j+20-1]++; break;
					 case 21: flavornum_day[t_gap_num][j+20-1]++; break;
					 case 22: flavornum_day[t_gap_num][j+20-1]++; break;
					 case 23: flavornum_day[t_gap_num][j+20-1]++; break;
					 case 24: flavornum_day[t_gap_num][j+20-1]++; break;
					 case 25: flavornum_day[t_gap_num][j+20-1]++; break;
					 default:  printf("虚拟机种类不止%d个\n",N);
				}
			}
		}
	}
	//printf("\n0\n");
	
	memcpy(flavornum_day_cheng, flavornum_day, sizeof(flavornum_day));
	int r = 0;
    for(unsigned int i=0;i<MAX_TRAIN_DAYS;i++)
    {
    	for(unsigned int j=0;j<N;j++)
    	{
        	r+=flavornum_day[i][j];
       	}
    }
    if(r==data_num)
	printf("读取train文件正确\n");
    else
	printf("读取train文件错误,错误数总数为%d\n",r);
	
}


void get_function(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num)
{
	get_train_flavor(data,t_flavor,data_num);
/*    	for( int i=0; i<data_num; i++)
    	{	
	printf ("%s\n",t_flavor[i]);
	//printf ("%s\n",t_flavor[i]);
	}
*/
	printf("\n********************时间戳信息*****************\n");
	get_train_timestamp(data,t_time_stamp,data_num);
    for( int i=0; i<11; i++)
    	{	
			printf ("%s\n",t_time_stamp[i]);
	//printf ("%s\n",t_flavor[i]);
		}
	printf("\n*************train文件虚拟机数量信息***************\n");
	get_train_flavornum(t_flavor,t_flavor_num,data_num);
    for( int i=0; i<N; i++)
    	{	
			printf ("flavor%d的个数为:%d\n", i+1,t_flavor_num[i]);
			//printf ("%s\n",t_flavor[i]);
		}
	//printf("%s\n",t_time_stamp[3]);
	printf("\n*************物理服务器核数***************\n");
	get_CPUcore_num(info,&core_num);
	printf ("%d\n",core_num);
	printf("\n*************物理服务器内存***************\n");
	get_RAM(info,&RAM);
	printf ("%d\n",RAM);
	printf("\n*************物理服务器硬盘***************\n");
	get_DISK(info,&DISK);
	printf ("%d\n",DISK);
	printf("\n*************需要预测虚拟机总类型数***************\n");
	get_input_flavorN(info,&i_flavorN);
	printf ("%d\n",i_flavorN);
	printf("\n*************待优化选项***************\n");
	if(get_opt_type(info)==CPU)
	{
		printf("the optimizaiton option is CPU\n");
		type_opti=CPU;
	}
	else if(get_opt_type(info)==MEM)
		{
	        printf("the optimizaiton option is MEM\n");
			type_opti=MEM;
		}
	     else 
	    	printf("INPUT FILE WORONG\n");
	    	
	//printf("%d\n",type_opti);
	printf("\n*************需预测虚拟机类型***************\n");
	get_input_flavor(info,i_flavor);
    	for(unsigned int i=0; i<i_flavorN; i++)
    	{	
			printf("%s\n",i_flavor[i]);
		}
/*
	get_input_flavorcore(info,i_flavor_core);
    	for(unsigned int i=0; i<i_flavorN; i++)
    	{	
	printf("%d\n",i_flavor_core[i]);
	}

	get_input_fRAM(info,i_flavor_RAM);
    	for(unsigned int i=0; i<i_flavorN; i++)
    	{	
	printf("%d\n",i_flavor_RAM[i]);
	}
*/
	printf("*************预测时间段信息***************\n");
	predict_day(info,predict_daystamp);


	
	printf("*************train,input时间间隔***************\n");
	i_gap=get_time_gap(predict_daystamp[0], predict_daystamp[1] );
    for(unsigned int i=0; i<2; i++)
	{	
		printf("%s\n",predict_daystamp[i]);
	}
	printf("%d\n",i_gap);
	t_gap=get_time_gap(t_time_stamp[0], t_time_stamp[data_num-1]);
	printf("%s\n",t_time_stamp[0]);
	printf("%s\n",t_time_stamp[data_num-1]);
	printf("%d\n",t_gap);
	
	get_flavornum_day(data_num);
//	printf("*************虚拟机使用情况***************\n");

//    for(int i=0;i<t_gap+1;i++)
//    {
//    	for(unsigned int j=0;j<N;j++)
//    	{
//        	printf("第%d天flavor%d的使用数量为: %d\n",i+1, j+1, flavornum_day[i][j]);
//       	}
//       	printf("\n");
//    }
	printf("*************自相关值统计***************\n");
	int Corr[t_gap+1];
	myAutoCorrelation(flavornum_day, Corr);
   	for(int c=0; c<t_gap+1; c++)
   		printf("间隔[%d]天的自相关值为%d\n", c, Corr[c]);
   	
}

