// Compile command for a linux system
// g++ -fopenmp -O2 -Wall -Wno-unused-result -Wno-unknown-pragmas -shared -o CC_CSSS_python_lib.so -fPIC CC_CSSS_python_lib.cc

// the NUMBER_OF_THREADS specifies how many cores should be utilized via OpenMP parallel computation
#define NUMBER_OF_THREADS 10

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <sys/resource.h>
#include <string.h>
#include <random>
#include <chrono>
#include <cstring>

using namespace std;

#define BAD_DATA_FLOAT -9999

// da prav dela error(..) - da prav displaya line number
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define FU __PRETTY_FUNCTION__
#define TT "\t"
#define tabt "\t"
#define ERRORIF(x) if (x) error(AT,FU, #x)

#include "CC_smoothing_on_sphere_python_lib.cc"


void calculate_smoothed_values_and_neighbourhood_area_sizes(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const uint32_t * const * const data_pointer,  vector <double> &f1_smoothed,  vector <double> &f2_smoothed,  vector <double> &neighbourhood_area_size)
	{
	vector <double> f1_x_area (number_of_points,0);
	vector <double> f2_x_area (number_of_points,0);

	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il=0; il < number_of_points; il++)
		{
		f1_x_area[il] = f1[il]*area_size[il];
		f2_x_area[il] = f2[il]*area_size[il];
		}


	vector <double> f1_x_area_sum (number_of_points,0);
	vector <double> f2_x_area_sum (number_of_points,0);
	vector <double> area_sum (number_of_points,0);


	// first loop to precalculate the partial sums of all points - can be parallelized
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		double partial_f1_x_area_sum=0;
		double partial_f2_x_area_sum=0;
		double partial_area_sum=0;

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		// terms from added points
		uint32_t start_index = 4;
		uint32_t end_index = start_index + data_pointer_for_point[2];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_f1_x_area_sum+=f1_x_area[index];
			partial_f2_x_area_sum+=f2_x_area[index];
			partial_area_sum+=area_size[index];
			}

		// terms from removed points
		start_index = end_index ;
		end_index = start_index + data_pointer_for_point[3];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_f1_x_area_sum-=f1_x_area[index];
			partial_f2_x_area_sum-=f2_x_area[index];
			partial_area_sum-=area_size[index];
			}

		f1_x_area_sum[data_pointer_for_point[0]] = partial_f1_x_area_sum;
		f2_x_area_sum[data_pointer_for_point[0]] = partial_f2_x_area_sum;
		area_sum[data_pointer_for_point[0]] = partial_area_sum;
		}

	// second loop to calculate the smoothed values - cannot be paralelized since it is iterative
	for (uint32_t il = 0; il < number_of_points; il++)
		{

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		uint32_t current_smoothing_point = data_pointer_for_point[0];
		uint32_t previous_smoothing_point = data_pointer_for_point[1];

		// check if this point has a reference point
		if (current_smoothing_point != previous_smoothing_point)
			{
			f1_x_area_sum[current_smoothing_point] += f1_x_area_sum[previous_smoothing_point];
			f2_x_area_sum[current_smoothing_point] += f2_x_area_sum[previous_smoothing_point];
			area_sum[current_smoothing_point] += area_sum[previous_smoothing_point];
			}

		if (area_sum[current_smoothing_point] > 0)
			{
			f1_smoothed[current_smoothing_point] = f1_x_area_sum[current_smoothing_point]/area_sum[current_smoothing_point];
			f2_smoothed[current_smoothing_point] = f2_x_area_sum[current_smoothing_point]/area_sum[current_smoothing_point];
			neighbourhood_area_size[current_smoothing_point]=area_sum[current_smoothing_point];
			}
		else
			{
			f1_smoothed[current_smoothing_point]=0;
			f2_smoothed[current_smoothing_point]=0;
			neighbourhood_area_size[current_smoothing_point]=area_sum[current_smoothing_point];
			}
		}

	// a third loop to set the missing data points to zero value - can be parallelized
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		if (area_size[il] == 0)
			{
			f1_smoothed[il]=0;
			f2_smoothed[il]=0;
			}
		}

	}


void calculate_CSSS2_value_with_gradient(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const uint32_t * const * const data_pointer,  double * const CSSS_value, double * const CSSS_gradient)
	{

 	vector <double> f1_smoothed (number_of_points,0);
 	vector <double> f2_smoothed (number_of_points,0);;
 	vector <double> neighbourhood_area_size (number_of_points,0);;

	//cout << number_of_points << endl;

	calculate_smoothed_values_and_neighbourhood_area_sizes(area_size, f1,  f2, number_of_points, data_pointer,  f1_smoothed,  f2_smoothed,  neighbourhood_area_size );

	double F=0;
	double G=0;
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		F+=area_size[il]*(f1_smoothed[il]-f2_smoothed[il])*(f1_smoothed[il]-f2_smoothed[il]);
		G+=area_size[il]*f1_smoothed[il]*f1_smoothed[il] + area_size[il]*f2_smoothed[il]*f2_smoothed[il];
		}
	*CSSS_value=1-F/G;

 	vector <double> Zj (number_of_points,0);
 	vector <double> Qj (number_of_points,0);

	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		Zj[il] = area_size[il]/neighbourhood_area_size[il]*(f1_smoothed[il] - f2_smoothed[il]);
		Qj[il] = area_size[il]/neighbourhood_area_size[il]*f2_smoothed[il];
		}

	vector <double> Pk_sum (number_of_points,0);
	vector <double> Rk_sum (number_of_points,0);

	// first loop to precalculate the partial sums of all points - can be parallelized
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		double partial_Pk_sum=0;
		double partial_Rk_sum=0;

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		// terms from added points
		uint32_t start_index = 4;
		uint32_t end_index = start_index + data_pointer_for_point[2];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_Pk_sum+=Zj[index];
			partial_Rk_sum+=Qj[index];
			}

		// terms from removed points
		start_index = end_index ;
		end_index = start_index + data_pointer_for_point[3];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_Pk_sum-=Zj[index];
			partial_Rk_sum-=Qj[index];
			}

		Pk_sum[data_pointer_for_point[0]] = partial_Pk_sum;
		Rk_sum[data_pointer_for_point[0]] = partial_Rk_sum;
		}

	// second loop to calculate the summed values - cannot be paralelized since it is iterative
	for (uint32_t il = 0; il < number_of_points; il++)
		{

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		uint32_t current_smoothing_point = data_pointer_for_point[0];
		uint32_t previous_smoothing_point = data_pointer_for_point[1];

		// check if this point has a reference point
		if (current_smoothing_point != previous_smoothing_point)
			{
			Pk_sum[current_smoothing_point] += Pk_sum[previous_smoothing_point];
			Rk_sum[current_smoothing_point] += Rk_sum[previous_smoothing_point];
			}
		}

	// calculate the gradients
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		CSSS_gradient[il] = 2*area_size[il]/(G*G)*(G*Pk_sum[il] + F*Rk_sum[il]);
		}


	}


extern "C" void calculate_CSSS2_value_with_gradient_ctypes(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const uint32_t * const * const data_pointer,  double * const CSSS_value, double * const CSSS_gradient)
	{
	auto begin4 = std::chrono::high_resolution_clock::now();

	calculate_CSSS2_value_with_gradient(area_size, f1,  f2, number_of_points, data_pointer,  CSSS_value, CSSS_gradient);

	cout << "----- CSSS calculation " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin4).count() * 1e-9 << " s" << endl;
 	}


void calculate_CSSSp_value_with_gradient(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const uint32_t * const * const data_pointer,  const double p, double * const CSSS_value, double * const CSSS_gradient)
	{

	// test if p is not smaller than one
	if (p < 1)
		error(AT,FU, "The p exponent cannot be smaller than 1 !");

 	vector <double> f1_smoothed (number_of_points,0);
 	vector <double> f2_smoothed (number_of_points,0);;
 	vector <double> neighbourhood_area_size (number_of_points,0);;

	//cout << number_of_points << endl;

	calculate_smoothed_values_and_neighbourhood_area_sizes(area_size, f1,  f2, number_of_points, data_pointer,  f1_smoothed,  f2_smoothed,  neighbourhood_area_size );

	double F=0;
	double G=0;
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		F+=area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), p);
		G+=area_size[il]*pow(fabs(f1_smoothed[il]), p) + area_size[il]*pow(fabs(f2_smoothed[il]), p);
		}
	*CSSS_value=1-F/G;

 	vector <double> Zj (number_of_points,0);
 	vector <double> Qj (number_of_points,0);

	const double pminusone = p - 1;
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		if ( f1_smoothed[il] > f2_smoothed[il])
			Zj[il] = area_size[il]/neighbourhood_area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), pminusone);
		else if ( f1_smoothed[il] < f2_smoothed[il])
			Zj[il] = - area_size[il]/neighbourhood_area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), pminusone);
		else
			Zj[il] = 0;

		if (f2_smoothed[il] > 0)
			Qj[il] = area_size[il]/neighbourhood_area_size[il]*pow(fabs(f2_smoothed[il]), pminusone);
		else if ( f2_smoothed[il] < 0 )
			Qj[il] = -area_size[il]/neighbourhood_area_size[il]*pow(fabs(f2_smoothed[il]), pminusone);
		else
			Qj[il] = 0;
		}

	vector <double> Pk_sum (number_of_points,0);
	vector <double> Rk_sum (number_of_points,0);

	// first loop to precalculate the partial sums of all points - can be parallelized
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		double partial_Pk_sum=0;
		double partial_Rk_sum=0;

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		// terms from added points
		uint32_t start_index = 4;
		uint32_t end_index = start_index + data_pointer_for_point[2];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_Pk_sum+=Zj[index];
			partial_Rk_sum+=Qj[index];
			}

		// terms from removed points
		start_index = end_index ;
		end_index = start_index + data_pointer_for_point[3];
		for (uint32_t ip = start_index; ip < end_index; ip++)
			{
			uint32_t index = data_pointer_for_point[ip];
			partial_Pk_sum-=Zj[index];
			partial_Rk_sum-=Qj[index];
			}

		Pk_sum[data_pointer_for_point[0]] = partial_Pk_sum;
		Rk_sum[data_pointer_for_point[0]] = partial_Rk_sum;
		}

	// second loop to calculate the summed values - cannot be paralelized since it is iterative
	for (uint32_t il = 0; il < number_of_points; il++)
		{

		const uint32_t * const data_pointer_for_point = data_pointer[il];

		uint32_t current_smoothing_point = data_pointer_for_point[0];
		uint32_t previous_smoothing_point = data_pointer_for_point[1];

		// check if this point has a reference point
		if (current_smoothing_point != previous_smoothing_point)
			{
			Pk_sum[current_smoothing_point] += Pk_sum[previous_smoothing_point];
			Rk_sum[current_smoothing_point] += Rk_sum[previous_smoothing_point];
			}
		}

	// calculate the gradients
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		CSSS_gradient[il] = p*area_size[il]/(G*G)*(G*Pk_sum[il] + F*Rk_sum[il]);
		}


	}

extern "C" void calculate_CSSSp_value_with_gradient_ctypes(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const uint32_t * const * const data_pointer,  const double p, double * const CSSS_value, double * const CSSS_gradient)
	{
	auto begin4 = std::chrono::high_resolution_clock::now();

	calculate_CSSSp_value_with_gradient(area_size, f1,  f2, number_of_points, data_pointer,  p, CSSS_value, CSSS_gradient);

	cout << "----- CSSS calculation " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin4).count() * 1e-9 << " s" << endl;

 	}

// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// kd-tree based gradient calulation
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------

void generate_fxareasize_and_areasize_Bounding_Box_data_for_kdtree_CSSS_calculation(const kdtree::KdTreeNode * const node, const double * const f1, const double * const f2, const double * const area_size,  vector <double> &f1_x_areasize_BB_data, vector <double> &f2_x_areasize_BB_data, vector <double> &area_size_BB_data)
	{
	size_t index = node->val.index;
	double f1_x_areasize_BB = f1[index]*area_size[index];
	double f2_x_areasize_BB = f2[index]*area_size[index];
	double areasize_BB = area_size[index];
	if(node->leftNode != nullptr)
		{
		generate_fxareasize_and_areasize_Bounding_Box_data_for_kdtree_CSSS_calculation(node->leftNode, f1, f2, area_size, f1_x_areasize_BB_data,  f2_x_areasize_BB_data, area_size_BB_data);
		f1_x_areasize_BB+= f1_x_areasize_BB_data[node->leftNode->val.index];
		f2_x_areasize_BB+= f2_x_areasize_BB_data[node->leftNode->val.index];
		areasize_BB+= area_size_BB_data[node->leftNode->val.index];
		}
	if(node->rightNode != nullptr)
		{
		generate_fxareasize_and_areasize_Bounding_Box_data_for_kdtree_CSSS_calculation(node->rightNode, f1, f2, area_size, f1_x_areasize_BB_data,  f2_x_areasize_BB_data, area_size_BB_data);
		f1_x_areasize_BB+= f1_x_areasize_BB_data[node->rightNode->val.index];
		f2_x_areasize_BB+= f2_x_areasize_BB_data[node->rightNode->val.index];
		areasize_BB+= area_size_BB_data[node->rightNode->val.index];
		}

	f1_x_areasize_BB_data[index] =f1_x_areasize_BB;
	f2_x_areasize_BB_data[index] =f2_x_areasize_BB;
	area_size_BB_data[index] =areasize_BB;
	}



void calculate_smoothed_values_and_neighbourhood_area_sizes_using_kdtree_smoothing_data(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const char * const data_pointer,  vector <double> &f1_smoothed,  vector <double> &f2_smoothed,  vector <double> &neighbourhood_area_size)
	{
	//auto begin = std::chrono::high_resolution_clock::now();

	const smoothing_data_for_the_kdtree_based_approach * const smoothing_data = (const smoothing_data_for_the_kdtree_based_approach * const) data_pointer;

	ERRORIF(number_of_points != smoothing_data->smoothing_data_for_point.size());

	vector <double> f1_x_area (number_of_points,0);
	vector <double> f2_x_area (number_of_points,0);

	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il=0; il < number_of_points; il++)
		{
		f1_x_area[il] = f1[il]*area_size[il];
		f2_x_area[il] = f2[il]*area_size[il];
		}

	vector <double> f1_x_areasize_BB_data (number_of_points,0);
	vector <double> f2_x_areasize_BB_data (number_of_points,0);
	vector <double> area_size_BB_data (number_of_points,0);

	generate_fxareasize_and_areasize_Bounding_Box_data_for_kdtree_CSSS_calculation(smoothing_data->kdtree.root, f1, f2, area_size,  f1_x_areasize_BB_data, f2_x_areasize_BB_data, area_size_BB_data);

	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il=0; il < number_of_points; il++)
		{
		double f1_x_areasize_temp=0;
		double f2_x_areasize_temp=0;
		double area_size_temp=0;

		const uint32_t * const smoothing_data_for_single_point =  smoothing_data->smoothing_data_for_point[il];
		uint32_t start_index = 2;
		uint32_t end_index = start_index + smoothing_data_for_single_point[0];
		for (uint32_t ip=start_index; ip < end_index; ip++)
			{
			f1_x_areasize_temp+=f1_x_areasize_BB_data[smoothing_data_for_single_point[ip]];
			f2_x_areasize_temp+=f2_x_areasize_BB_data[smoothing_data_for_single_point[ip]];
			area_size_temp+=area_size_BB_data[smoothing_data_for_single_point[ip]];
			}
		start_index = end_index;
		end_index = start_index + smoothing_data_for_single_point[1];
		for (uint32_t ip=start_index; ip < end_index; ip++)
			{
			f1_x_areasize_temp+=f1_x_area[smoothing_data_for_single_point[ip]];
			f2_x_areasize_temp+=f2_x_area[smoothing_data_for_single_point[ip]];
			area_size_temp+=area_size[smoothing_data_for_single_point[ip]];
			}

		neighbourhood_area_size[il] = area_size_temp;
		if (area_size_temp > 0)
			{
			f1_smoothed[il] = f1_x_areasize_temp / area_size_temp;
			f2_smoothed[il] = f2_x_areasize_temp / area_size_temp;
			}
		else
			{
			f1_smoothed[il] = 0;
			f2_smoothed[il] = 0;
			}

		}

	//cout << "----- smoothing " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;
	}




void generate_Zj_and_Qj_and_Bounding_Box_data_for_kdtree_CSSS_calculation(const kdtree::KdTreeNode * const node, const vector <double> &Zj, const vector <double> & Qj,  vector <double> &Zj_BB_data, vector <double> &Qj_BB_data)
	{
	size_t index = node->val.index;
	double Zj_BB = Zj[index];
	double Qj_BB = Qj[index];
	if(node->leftNode != nullptr)
		{
		generate_Zj_and_Qj_and_Bounding_Box_data_for_kdtree_CSSS_calculation(node->leftNode, Zj, Qj, Zj_BB_data,  Qj_BB_data);
		Zj_BB+= Zj_BB_data[node->leftNode->val.index];
		Qj_BB+= Qj_BB_data[node->leftNode->val.index];
		}
	if(node->rightNode != nullptr)
		{
		generate_Zj_and_Qj_and_Bounding_Box_data_for_kdtree_CSSS_calculation(node->rightNode, Zj, Qj, Zj_BB_data,  Qj_BB_data);
		Zj_BB+= Zj_BB_data[node->rightNode->val.index];
		Qj_BB+= Qj_BB_data[node->rightNode->val.index];
		}

	Zj_BB_data[index] =Zj_BB;
	Qj_BB_data[index] =Qj_BB;
	}


void calculate_CSSSp_value_with_gradient_using_kdtree_smoothing_data(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points, const char * const data_pointer,  const double p, double * const CSSS_value, double * const CSSS_gradient)
	{

	// test if p is not smaller than one
	if (p < 1)
		error(AT,FU, "The p exponent cannot be smaller than 1 !");

 	vector <double> f1_smoothed (number_of_points,0);
 	vector <double> f2_smoothed (number_of_points,0);;
 	vector <double> neighbourhood_area_size (number_of_points,0);;

	//cout << number_of_points << endl;

	calculate_smoothed_values_and_neighbourhood_area_sizes_using_kdtree_smoothing_data(area_size, f1,  f2, number_of_points, data_pointer,  f1_smoothed,  f2_smoothed,  neighbourhood_area_size );

	double F=0;
	double G=0;
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		F+=area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), p);
		G+=area_size[il]*pow(fabs(f1_smoothed[il]), p) + area_size[il]*pow(fabs(f2_smoothed[il]), p);
		}
	*CSSS_value=1-F/G;

 	vector <double> Zj (number_of_points,0);
 	vector <double> Qj (number_of_points,0);

	const double pminusone = p - 1;
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		if ( f1_smoothed[il] > f2_smoothed[il])
			Zj[il] = area_size[il]/neighbourhood_area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), pminusone);
		else if ( f1_smoothed[il] < f2_smoothed[il])
			Zj[il] = - area_size[il]/neighbourhood_area_size[il]*pow(fabs(f1_smoothed[il]-f2_smoothed[il]), pminusone);
		else
			Zj[il] = 0;

		if (f2_smoothed[il] > 0)
			Qj[il] = area_size[il]/neighbourhood_area_size[il]*pow(fabs(f2_smoothed[il]), pminusone);
		else if ( f2_smoothed[il] < 0 )
			Qj[il] = -area_size[il]/neighbourhood_area_size[il]*pow(fabs(f2_smoothed[il]), pminusone);
		else
			Qj[il] = 0;
		}



	vector <double> Zj_BB_data (number_of_points,0);
	vector <double> Qj_BB_data (number_of_points,0);

	const smoothing_data_for_the_kdtree_based_approach * const smoothing_data = (const smoothing_data_for_the_kdtree_based_approach * const) data_pointer;

	generate_Zj_and_Qj_and_Bounding_Box_data_for_kdtree_CSSS_calculation(smoothing_data->kdtree.root, Zj, Qj, Zj_BB_data, Qj_BB_data);


	vector <double> Pk_sum (number_of_points,0);
	vector <double> Rk_sum (number_of_points,0);

	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il=0; il < number_of_points; il++)
		{
		double Pk_sum_temp=0;
		double Rk_sum_temp=0;

		const uint32_t * const smoothing_data_for_single_point =  smoothing_data->smoothing_data_for_point[il];
		uint32_t start_index = 2;
		uint32_t end_index = start_index + smoothing_data_for_single_point[0];
		for (uint32_t ip=start_index; ip < end_index; ip++)
			{
			Pk_sum_temp+=Zj_BB_data[smoothing_data_for_single_point[ip]];
			Rk_sum_temp+=Qj_BB_data[smoothing_data_for_single_point[ip]];
			}
		start_index = end_index;
		end_index = start_index + smoothing_data_for_single_point[1];
		for (uint32_t ip=start_index; ip < end_index; ip++)
			{
			Pk_sum_temp+=Zj[smoothing_data_for_single_point[ip]];
			Rk_sum_temp+=Qj[smoothing_data_for_single_point[ip]];
			}

		Pk_sum[il] = Pk_sum_temp;
		Rk_sum[il] = Rk_sum_temp;
		}


	// calculate the gradients
	#pragma omp parallel for num_threads(NUMBER_OF_THREADS)
	for (uint32_t il = 0; il < number_of_points; il++)
		{
		CSSS_gradient[il] = p*area_size[il]/(G*G)*(G*Pk_sum[il] + F*Rk_sum[il]);
		}


	}

extern "C" void calculate_CSSSp_value_with_gradient_using_kdtree_smoothing_data_ctypes(const double * const area_size, const double * const f1,  const double * const f2, const size_t number_of_points,  const char * const data_pointer,  const double p, double * const CSSS_value, double * const CSSS_gradient)
	{
	auto begin4 = std::chrono::high_resolution_clock::now();

	calculate_CSSSp_value_with_gradient_using_kdtree_smoothing_data(area_size, f1,  f2, number_of_points, data_pointer,  p, CSSS_value, CSSS_gradient);

	cout << "----- CSSS calculation " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin4).count() * 1e-9 << " s" << endl;

 	}
