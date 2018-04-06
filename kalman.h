#ifndef __KALMAN_H__
#define __KALMAN_H__

// kalman_filter.m
// Q和R分别代表噪声和协方差
// 可以加控制量u(k)，可以进一步实现扩展卡尔曼滤波(EKF)、无迹卡尔曼滤波(UKF)等各种算法
//   来应对非线性、非高斯等各种情况。只是对于我来说，现在这样就暂时够用了。
// flavor_type 表示虚拟机类型 从0开始
extern void kalman(int * flavornum_day_f, int data_len, double * X,
			  double Q, double R, double P0);
			
			
#endif
