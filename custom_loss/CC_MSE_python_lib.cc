// Compile command for a linux system
// g++ -fopenmp -O2 -Wall -Wno-unused-result -Wno-unknown-pragmas -shared -o CC_MSE_python_lib.so -fPIC CC_MSE_python_lib.cc

#include <iostream>

using namespace std;

extern "C"  void calculate_MSE_with_gradient_ctypes(const float * const values1, const float * const values2, const float * const area_size, const size_t number_of_points, float * const MSE_value, float * const MSE_gradient)
	{
	float MSE_sum=0;
	float area_size_sum=0;

	for (size_t il=0; il < number_of_points; il++)
		{
		MSE_sum+=area_size[il]*(values1[il]-values2[il])*(values1[il]-values2[il]);
		area_size_sum+=area_size[il];
		}

	for (size_t il=0; il < number_of_points; il++)
		{
		MSE_gradient[il]=-area_size[il]*2.0*(values1[il]-values2[il])/area_size_sum;
		}

	*MSE_value=MSE_sum/area_size_sum;
	}



