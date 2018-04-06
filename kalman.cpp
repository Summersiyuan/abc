#include "predict.h"
#include "get_info.h"
#include "average.h"
#include "function.h"
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

#ifdef _DEBUG
#define PRINT   printf
#else
#define PRINT(...)
#endif


//Q和R分别代表噪声和协方差
void kalman(int * flavornum_day_f, int data_len, double * X,
			 double Q, double R, double P0)
{
	double K[MAX_TRAIN_DAYS], P[MAX_TRAIN_DAYS];
	X[0]=flavornum_day_f[0];
	P[0]=P0;
	for(int i=1; i<data_len+1; i++)
	{
		K[i] = P[i-1] / (P[i-1] + R);
    	X[i] = X[i-1] + K[i] * (flavornum_day_f[i] - X[i-1]);
    	P[i] = P[i-1] - K[i] * P[i-1] + Q;
	}
	//return X;
}
