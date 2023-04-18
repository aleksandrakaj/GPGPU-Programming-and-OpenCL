#include "CL/cl.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"

const char* kernelstring =
"__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int m, int n, int k) {"
"int globalRow = get_global_id(0);"
"int globalColumn = get_global_id(1);"
"float tmp = 0;"
"for (int j = 0; j < k; ++j)"
"{"
"tmp += A[j * m + globalRow] * B[globalColumn * k + j];"
"}"
"C[globalColumn * m + globalRow] = tmp;"
"}";

void main() {

    size_t valueSize;
    char* value;

    cl_uint maxComputeUnits;
    size_t maxWorkGroupSize;
    cl_ulong maxGlobalMemorySize;
    cl_ulong maxLocalMemorySize;

    cl_event event = NULL;

    // get platform
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);


    // get device
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);



    // print device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);


    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf(" Parallel compute units: %d\n", (int)maxComputeUnits);


    // print max work group size
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    printf(" Max work group size: %d\n", (int)maxWorkGroupSize);


    // print max global memory size
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(maxGlobalMemorySize), &maxGlobalMemorySize, NULL);
    printf(" Max global memory size: %d\n", (int)maxGlobalMemorySize);


    // print max local memory size
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(maxLocalMemorySize), &maxLocalMemorySize, NULL);
    printf(" Max local memory size: %d\n", (int)maxLocalMemorySize);


    int m, n, k, num;
    printf("Type in m, n and k and num: \n");
    scanf_s("%d", &m);
    scanf_s("%d", &n);
    scanf_s("%d", &k);
    scanf_s("%d", &num);

    // set seed for rand()
    srand(2014);

    //Allocate host memory for matrices A and B and C
    float* arr_A = (float*)malloc(m * k * sizeof(float));
    float* arr_B = (float*)malloc(k * n * sizeof(float));
    float* arr_C = (float*)malloc(m * n * sizeof(float));

    //Initialize host memory
    for (int i = 0; i < m * k; i++) { arr_A[i] = (float)rand(); }
    for (int i = 0; i < k * n; i++) { arr_B[i] = (float)rand(); }
    for (int i = 0; i < m * n; i++) { arr_C[i] = 0.0; }

    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    cl_command_queue commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);



    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, m * k * sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, k * n * sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, m * n * sizeof(float), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(commands, bufA, CL_TRUE, 0, m * k * sizeof(float), arr_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(commands, bufB, CL_TRUE, 0, k * n * sizeof(float), arr_B, 0, NULL, NULL);
    clEnqueueWriteBuffer(commands, bufC, CL_TRUE, 0, m * n * sizeof(float), arr_C, 0, NULL, NULL);

    //Launch OpenCL kernel
    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = 32;
    localWorkSize[1] = 32;
    globalWorkSize[0] = m;
    globalWorkSize[1] = n;

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufC);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&m);
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&n);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&k);

    double avgTime = 0.0;
    double* times = (double*)malloc(num * sizeof(double));


    cl_ulong time_start;
    cl_ulong time_end;


    for (int i = 0; i < num; i++) {

        clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);

        clWaitForEvents(1, &event);
        cl_int err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        if (err != CL_SUCCESS) {
            printf("error %d", err);
        }

        avgTime += (time_end - time_start) / 1000000.0;
        times[i] = (double)(time_end - time_start) / 1000000.0;
    }


    avgTime = (avgTime / (double)num);
    printf("Average time: %f miliseconds\n", avgTime);

    double sigma = 0.0;
    for (int i = 0; i < num; i++) {
        sigma += (times[i] - avgTime) * (times[i] - avgTime);

    }
    sigma = (double)sqrt(sigma / (double)num);
    printf("Standard deviation: %0.10f", sigma);

    //Retrieve result from device
    clEnqueueReadBuffer(commands, bufC, CL_TRUE, 0, m * n * sizeof(float), arr_C, 0, NULL, NULL);


    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Clean-up OpenCL 
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Free the host memory objects
    free(arr_A);
    free(arr_B);
    free(arr_C);



}