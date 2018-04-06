#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "function.h"
#include "least_square_method.h"
#include "lstm.h"
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

int flavor[N][3]=  //定义虚拟机属性
{
	//{虚拟机预测个数,内核数(个),内存大小(GB)}
	{0,1,1},
	{0,1,2},
	{0,1,4},
	{0,2,2},
	{0,2,4},
	{0,2,8},
	{0,4,4},
	{0,4,8},
	{0,4,16},
	{0,8,8},
	{0,8,16},
	{0,8,32},
	{0,16,16},
	{0,16,32},
	{0,16,64}

};


//ÄãÒªÍê³ÉµÄ¹ŠÄÜ×ÜÈë¿Ú

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{

	get_function(info, data, data_num); //获取信息总函数
	
	//average( t_flavor_num, i_flavor, flavor, i_flavorN, data_num);//平均数预测
	
	//least_square_mul(i_flavorN); //曲线+平均数预测
	for(unsigned int f=0; f<i_flavorN; f++)
	lstm(predict_flavor[f], t_gap, i_gap);
	
	for(unsigned int i=0; i<N; i++)
    {	
		printf("预测虚拟机flavor%d的个数为%d\n",i+1,flavor[i][0]);
	}
	printf("\n*************需要预测虚拟机***************\n");
		for(unsigned int i=0; i<i_flavorN;i++)
	{
		printf("flavor%d\n",predict_flavor[i]);
	}
	
	//printf("%d\n",type_opti);
	//printf("%s\n",filename);
	int flavor_i[N][3]={0};
	memcpy(flavor_i, flavor, sizeof(flavor));
	put(core_num, RAM, flavor_i, type_opti); //放置虚拟机任务函数
	
	int sum_flavor=0; //预测虚拟机总数
		for(unsigned int i=0; i<i_flavorN; i++)
    {	
    	sum_flavor+=flavor[predict_flavor[i]-1][0];
		//printf("预测虚拟机flavor%d的个数为%d\n",i+1,flavor[i][0]);
	}
	//printf("%s\n",filename);	
	// ÐèÒªÊä³öµÄÄÚÈÝ
	//char * result_file = (char *)"18\n\n0 8 0 20";
	printf("\n*************output文件信息***************\n");
	//char result_file[] = "18\n\n0 8 0 20";
	unsigned int k,j=0;//计算输出文件行数
	k=4+i_flavorN+number_server;
	//printf("%d\n",k);
	int z=0;
	char buff_line[k*MAX_LINE_LEN]; //输入文件信息流
	
	//buff_line = (char *)malloc(1000);
	for(unsigned int i=0; i<k; i++)
	{
		if(i==0)
		{
			
			z+=sprintf(buff_line, "%d\n", sum_flavor );
			//*buff_line= *result_file[0];
		}
		else if(i>0&&i<i_flavorN+1)
		{
			//result_file[i] = (char *)malloc(MAX_LINE_LEN + 2);
			int n =predict_flavor[i-1];
			z+=sprintf(buff_line+z, "flavor%d %d\n", predict_flavor[i-1], flavor[n-1][0]); 
		}
		else if(i==i_flavorN+1)
		{
			//result_file[i] = (char *)malloc(MAX_LINE_LEN + 2);
			z+=sprintf(buff_line+z, "\n" );
		}
		else if(i==i_flavorN+2)
		{
			//result_file[i] = (char *)malloc(MAX_LINE_LEN + 2);
			z+=sprintf(buff_line+z, "%d\n", number_server+1);
		}
		else if(i>i_flavorN+2)
		{
			j++;
			z+=sprintf(buff_line+z, "%d ", j);
			//result_file[i] = (char *)malloc(MAX_LINE_LEN + 2);
			for(unsigned int m=0; m<i_flavorN; m++)
			{
				if(output_i[j-1][predict_flavor[m]-1]!=0)
				{
					if(m!=i_flavorN-1)
						z+=sprintf(buff_line+z, "flavor%d %d ", predict_flavor[m], output_i[j-1][predict_flavor[m]-1]);
					else if(m==i_flavorN-1)
						z+=sprintf(buff_line+z, "flavor%d %d", predict_flavor[m], output_i[j-1][predict_flavor[m]-1]);
				}
			}
			if(i!=k-1)
			z+=sprintf(buff_line+z, "\n");
		}
	
		//z  = sprintf( result_file[i],    "   String:    %s\n", s ); // C4996
   		//z += sprintf( buff_line + z, "%s", result_file[i] ); // C4996
	}
	// Ö±œÓµ÷ÓÃÊä³öÎÄŒþµÄ·œ·šÊä³öµœÖž¶šÎÄŒþÖÐ(psÇë×¢ÒâžñÊœµÄÕýÈ·ÐÔ£¬Èç¹ûÓÐœâ£¬µÚÒ»ÐÐÖ»ÓÐÒ»žöÊýŸÝ£»µÚ¶þÐÐÎª¿Õ£»µÚÈýÐÐ¿ªÊŒ²ÅÊÇŸßÌåµÄÊýŸÝ£¬ÊýŸÝÖ®ŒäÓÃÒ»žö¿Õžñ·Öžô¿ª)
	printf("%s\n",buff_line);
	//printf("%s\n",filename);
	write_result(buff_line, filename);
    release_buff(t_flavor, data_num);
	//release_buff(data, data_line_num);
}

