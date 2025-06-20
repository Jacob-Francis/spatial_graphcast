import numpy as np
import os
import ctypes
from ctypes import *

# search for the PAD C++ shared library file (PAD_on_sphere_Cxx_shared_library.so) in the same folder
# libc = ctypes.CDLL(os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))+ os.path.sep + "CC_CSSS_python_lib.so")
libc = ctypes.CDLL(
    os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
    + os.path.sep
    + "CC_CSSS_python_lib.so"
)

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.free_mem_double_array.argtypes = [ctypes.POINTER(ctypes.c_double)]
libc.free_mem_double_array.restype = None

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.generate_smoothing_data_for_the_overlap_detection_and_write_it_to_disk_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ctypes.c_size_t,
    ND_POINTER_1D,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
# libc.smooth_field_using_KdTree_ctypes.restype = ctypes.POINTER(ctypes.c_double)
libc.generate_smoothing_data_for_the_overlap_detection_and_write_it_to_disk_ctypes.restype = (
    None
)


def generate_smoothing_data_for_the_overlap_detection_based_approach_and_write_it_to_the_disk(
    lat, lon, smoothing_kernel_radius_in_metres, smoothing_data_folder
):

    # convert array to np.float64 and contiguousarray  - this format is required for the interation with the C++ code
    lat = np.ascontiguousarray(lat, dtype=np.float64)
    lon = np.ascontiguousarray(lon, dtype=np.float64)

    if np.ndim(smoothing_kernel_radius_in_metres) > 1:
        print(
            'ERROR: the smoothing_kernel_radius_in_metres needs to bo a single value or a 1D array. Returning "None" as result!'
        )
        return None

    # if smoothing_kernel_radius_in_metres is a single value convert it to an array
    if np.ndim(smoothing_kernel_radius_in_metres) == 0:
        smoothing_kernel_radius_in_metres = np.ascontiguousarray(
            [smoothing_kernel_radius_in_metres], dtype=np.float64
        )

    # convert array to np.float64 and contiguousarray  - this format is required for the interaction with the C++ code
    smoothing_kernel_radius_in_metres = np.ascontiguousarray(
        smoothing_kernel_radius_in_metres, dtype=np.float64
    )

    # calculate the smoothed values
    libc.generate_smoothing_data_for_the_overlap_detection_and_write_it_to_disk_ctypes(
        lat,
        lon,
        len(lat),
        smoothing_kernel_radius_in_metres,
        len(smoothing_kernel_radius_in_metres),
        smoothing_data_folder,
        0,
    )


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.read_smoothing_data_from_binary_file_ctypes.argtypes = [
    ctypes.c_char_p,
    ctypes.c_double,
    ctypes.c_size_t,
]
libc.read_smoothing_data_from_binary_file_ctypes.restype = ctypes.c_void_p


def read_smoothing_data_from_binary_file(
    smoothing_data_folder, smoothing_kernel_radius_in_metres, number_of_points
):

    if np.ndim(smoothing_kernel_radius_in_metres) != 0:
        print(
            'ERROR: the smoothing_kernel_radius_in_metres needs to bo a single value in the read_smoothing_data_from_binary_file function. Returning "None" as result!'
        )
        return None

    smoothing_data_pointer = libc.read_smoothing_data_from_binary_file_ctypes(
        smoothing_data_folder, smoothing_kernel_radius_in_metres, number_of_points
    )
    return smoothing_data_pointer


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.free_smoothing_data_memory_ctypes.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.free_smoothing_data_memory_ctypes.restype = None


def free_smoothing_data_memory(smoothing_data_pointer, number_of_points):
    libc.free_smoothing_data_memory_ctypes(smoothing_data_pointer, number_of_points)
    smoothing_data_pointer = None


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.smooth_field_using_overlap_detection_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ND_POINTER_1D,
]
libc.smooth_field_using_overlap_detection_ctypes.restype = None


def smooth_field_using_overlap_detection(area_size, f, smoothing_data_pointer):

    # convert array to np.float64 and contiguousarray  - this format is required for the interation with the C++ code
    area_size = np.ascontiguousarray(area_size, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)

    # reserve memory for the outputed smoothed field
    f_smoothed = np.ascontiguousarray(np.zeros(len(f), dtype=np.float64))

    # calculate the smoothed values
    libc.smooth_field_using_overlap_detection_ctypes(
        area_size, f, len(f), smoothing_data_pointer, f_smoothed
    )

    return f_smoothed


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


libc.calculate_CSSS2_value_with_gradient_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    ctypes.c_size_t,
    ctypes.c_void_p,
    POINTER(c_double),
    ND_POINTER_1D,
]
libc.calculate_CSSS2_value_with_gradient_ctypes.restype = None


def calculate_CSSS2_value_with_gradient(
    values1, values2, area_size, smoothing_data_pointer
):

    CSSS_gradient = np.ascontiguousarray(np.zeros(len(values1), dtype=np.float64))
    c_CSSS_value = c_double()

    libc.calculate_CSSS2_value_with_gradient_ctypes(
        area_size,
        values1,
        values2,
        len(values1),
        smoothing_data_pointer,
        byref(c_CSSS_value),
        CSSS_gradient,
    )

    CSSS_value = c_CSSS_value.value

    return [CSSS_value, CSSS_gradient]


# -----------------------------------------------------------------------------------------------------
#                                 BACTHED VERSION OF THE CSSS2 LOSS FUNCTION
# -----------------------------------------------------------------------------------------------------
ND_POINTER_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")


libc.batch_calculate_css2_val_and_grad.argtypes = [
    ND_POINTER_2D,
    ND_POINTER_2D,
    ND_POINTER_2D,
    ctypes.POINTER(ctypes.c_int64),  # dimensions pointer
    ctypes.c_size_t,
    ctypes.c_void_p,
    ND_POINTER_1D,
    ND_POINTER_2D,
]
libc.batch_calculate_css2_val_and_grad.restype = None


def batch_calculate_CSSS2_value_with_gradient(
    values1, values2, area_size, smoothing_data_pointer
):

    batch_c_CSSS_value = np.ascontiguousarray(
        np.zeros(values1.shape[0]), dtype=np.float64
    )

    BATCH_CSSS_gradient = np.ascontiguousarray(np.zeros_like(values1), dtype=np.float64)
    dims = values1.shape

	# Convert to a ctypes array
    DimsArrayType = ctypes.c_int64 * len(dims)
    dims_ctypes = DimsArrayType(*dims)
    
    assert area_size.flags['C_CONTIGUOUS']
    assert values1.flags['C_CONTIGUOUS']
    assert values2.flags['C_CONTIGUOUS']

    libc.batch_calculate_css2_val_and_grad(
        area_size,
        values1,
        values2,
        dims_ctypes,
        len(values1.shape),
        smoothing_data_pointer,
        batch_c_CSSS_value,
        BATCH_CSSS_gradient,
    )

    return [batch_c_CSSS_value, BATCH_CSSS_gradient]
