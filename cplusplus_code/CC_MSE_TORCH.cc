// Compile command for a linux system
// g++ -fopenmp -O2 -Wall -Wno-unused-result -Wno-unknown-pragmas -shared -o CC_MSE_python_lib.so -fPIC CC_MSE_python_lib.cc
// g++ -std=c++17 -fopenmp -O2 -Wall -Wno-unused-result -Wno-unknown-pragmas -shared -fPIC \
  -o CC_MSE_LOWERING.so CC_MSE_LOWERING.cc

#include <iostream>

using namespace std;

extern "C"  void calculate_MSE_ctypes(
    const double * const values1,
    const double * const values2,
    const double * const area_size,
    const size_t number_of_points,
    double * const MSE_value
    )
	{
	double MSE_sum=0;
	double area_size_sum=0;

	for (size_t il=0; il < number_of_points; il++)
		{
		MSE_sum+=area_size[il]*(values1[il]-values2[il])*(values1[il]-values2[il]);
		area_size_sum+=area_size[il];
		}

	*MSE_value=MSE_sum/area_size_sum;
	}

extern "C" void calculate_MSE_gradient_ctypes(
    const double* const values1,
    const double* const values2,
    const double* const area_size,
    const size_t number_of_points,
    double* const MSE_gradient)
{
    double area_size_sum = 0.0;

    // First compute area_size_sum
    for (size_t il = 0; il < number_of_points; il++)
    {
        area_size_sum += area_size[il];
    }

    // Then compute gradient values
    for (size_t il = 0; il < number_of_points; il++)
    {
        MSE_gradient[il] = -area_size[il] * 2.0 * (values1[il] - values2[il]) / area_size_sum;
    }
}


// #include <functional>
// #include <numeric>
// #include <utility>

// #include "xla/ffi/api/c_api.h"
// #include "xla/ffi/api/ffi.h"

// namespace ffi = xla::ffi;

// // A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// template <ffi::DataType T>
// std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
//   auto dims = buffer.dimensions();
//   if (dims.size() == 0) {
//     return std::make_pair(0, 0);
//   }
//   return std::make_pair(buffer.element_count(), dims.back());
// }

// ffi::Error MseGradImpl(ffi::Buffer<ffi::F64> values1,
//                    ffi::Buffer<ffi::F64> values2,
//                    ffi::Buffer<ffi::F64> area_size,
//                    ffi::ResultBuffer<ffi::F64> MSE_gradient) {
//   auto [totalSize, lastDim] = GetDims(values1);

//   if (lastDim == 0) {
//     return ffi::Error::InvalidArgument("Input must be a non-empty array");
//   }

//   if (values2.element_count() != values1.element_count() ||
//       area_size.element_count() != values1.element_count()) {
//     return ffi::Error::InvalidArgument("All input buffers must have the same number of elements");
//   }

//   for (int n = 0; n < totalSize; n += lastDim) {
//     calculate_MSE_gradient_ctypes(
//       &(values1.typed_data()[n]), 
//       &(values2.typed_data()[n]),
//       &(area_size.typed_data()[n]),
//       lastDim,
//       &(MSE_gradient->typed_data()[n])     
//     );

//     }
  
//   return ffi::Error::Success();
// }


// // Register the FFI handler with XLA
// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     MseGrad, MseGradImpl,
//     ffi::Ffi::Bind()
//         .Arg<ffi::Buffer<ffi::F64>>()  // values1
//         .Arg<ffi::Buffer<ffi::F64>>()  // values2
//         .Arg<ffi::Buffer<ffi::F64>>()  // area_size
//         .Ret<ffi::Buffer<ffi::F64>>()  // mse_value (one value per batch)
// );


// ffi::Error MseImpl(ffi::Buffer<ffi::F64> values1,
//                    ffi::Buffer<ffi::F64> values2,
//                    ffi::Buffer<ffi::F64> area_size,
//                    ffi::ResultBuffer<ffi::F64> mse_value) {
//   auto [totalSize, lastDim] = GetDims(values1);

//   if (lastDim == 0) {
//     return ffi::Error::InvalidArgument("Input must be a non-empty array");
//   }

//   if (values2.element_count() != values1.element_count() ||
//       area_size.element_count() != values1.element_count()) {
//     return ffi::Error::InvalidArgument("All input buffers must have the same number of elements");
//   }

//   for (int n = 0; n < totalSize; n += lastDim) {
//     double mse_scalar = 0.0;

//     calculate_MSE_ctypes(
//       &(values1.typed_data()[n]), 
//       &(values2.typed_data()[n]),
//       &(area_size.typed_data()[n]),
//       lastDim,
//       &mse_scalar   
//     );
//     mse_value->typed_data()[n/lastDim] = mse_scalar;
//     }

//   return ffi::Error::Success();
// }


// // Register the FFI handler with XLA
// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     Mse, MseImpl,
//     ffi::Ffi::Bind()
//         .Arg<ffi::Buffer<ffi::F64>>()  // values1
//         .Arg<ffi::Buffer<ffi::F64>>()  // values2
//         .Arg<ffi::Buffer<ffi::F64>>()  // area_size
//         .Ret<ffi::Buffer<ffi::F64>>()  // mse_value (one value per batch)
// );