package lym;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2016 Marco Hutter - http://www.jcuda.org
 */

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

/**
 * An example showing how to use the NVRTC (NVIDIA Runtime Compiler) API
 * to compile CUDA kernel code at runtime.
 */
//javac -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcVectorAdd.java
//java -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcVectorAdd

public class JNvrtcVectorAdd
{
    /**
     * The source code of the program that will be compiled at runtime:
     * A simple vector addition kernel. 
     * 
     * Note: The function should be declared as  
     * extern "C"
     * to make sure that it can be found under the given name.
     */
    private static String programSourceCode = 
        "extern \"C\"" + "\n" +
        "__global__ void add(int n, float *a, float *b, float *sum)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        sum[i] = a[i] + b[i];" + "\n" +
        "    }" + "\n" +
        "}" + "\n";
    
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        
        // Use the NVRTC to create a program by compiling the source code
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(
            program, programSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation log:\n" + programLog[0]);        
        
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");


        // Continue with some basic setup for the vector addition itself:

        // Allocate and fill the host input data
        int numElements = 256 * 100;
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = (float)i;
            hostInputB[i] = (float)i;
        }

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
            numElements * Sizeof.FLOAT);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
            numElements * Sizeof.FLOAT);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{numElements}),
            Pointer.to(deviceInputA),
            Pointer.to(deviceInputB),
            Pointer.to(deviceOutput)
        );

        
        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        int blockSizeX = 256;
        int gridSizeX = (numElements + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numElements * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numElements; i++)
        {
            float expected = i+i;
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);
        
    }
}

/**
CUDA_PATH="/usr/local/cuda-8.0/"
g++ saxpy_rtc.cpp -o saxpy_rtc -I $CUDA_PATH/include -L $CUDA_PATH/lib64  -lnvrtc -lcuda  -Wl,-rpath,$CUDA_PATH/lib64

A. Example: SAXPY
A.1. Code (saxpy.cpp)

#include <nvrtc.h>
#include <cuda.h>
#include <iostream>

#define NUM_THREADS 128
#define NUM_BLOCKS 32
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const char *saxpy = "                                           \n\
extern \"C\" __global__                                         \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)   \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    out[tid] = a * x[tid] + y[tid];                             \n\
  }                                                             \n\
}                                                               \n";

int main()
{
  // Create an instance of nvrtcProgram with the SAXPY code string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog,         // prog
                       saxpy,         // buffer
                       "saxpy.cu",    // name
                       0,             // numHeaders
                       NULL,          // headers
                       NULL));        // includeNames
  // Compile the program for compute_20 with fmad disabled.
  const char *opts[] = {"--gpu-architecture=compute_20",
                        "--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  2,     // numOptions
                                                  opts); // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
    exit(1);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  // Load the generated PTX and get a handle to the SAXPY kernel.
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "saxpy"));
  // Generate input for execution, and create output buffers.
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);
  float a = 5.1f;
  float *hX = new float[n], *hY = new float[n], *hOut = new float[n];
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }
  CUdeviceptr dX, dY, dOut;
  CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));
  // Execute SAXPY.
  void *args[] = { &a, &dX, &dY, &dOut, &n };
  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
                   NUM_BLOCKS, 1, 1,    // grid dim
                   NUM_THREADS, 1, 1,   // block dim
                   0, NULL,             // shared mem and stream
                   args, 0));           // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());
  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
  for (size_t i = 0; i < n; ++i) {
    std::cout << a << " * " << hX[i] << " + " << hY[i]
              << " = " << hOut[i] << '\n';
  }
  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(dX));
  CUDA_SAFE_CALL(cuMemFree(dY));
  CUDA_SAFE_CALL(cuMemFree(dOut));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] hX;
  delete[] hY;
  delete[] hOut;
  return 0;
}

A.2. Build Instruction
Assuming the environment variable CUDA_PATH points to CUDA Toolkit installation directory, build this example as:
Windows:
cl.exe saxpy.cpp /Fesaxpy ^
  /I "%CUDA_PATH%"\include ^
  "%CUDA_PATH%"\lib\x64\nvrtc.lib "%CUDA_PATH%"\lib\x64\cuda.lib
Linux:
g++ saxpy.cpp -o saxpy \
  -I $CUDA_PATH/include \
  -L $CUDA_PATH/lib64 \
  -lnvrtc -lcuda \
  -Wl,-rpath,$CUDA_PATH/lib64
Mac OS X:
clang++ saxpy.cpp -o saxpy \
  -I $CUDA_PATH/include \
  -L $CUDA_PATH/lib \
  -lnvrtc -framework CUDA \
  -Wl,-rpath,$CUDA_PATH/lib

Read more at: http://docs.nvidia.com/cuda/nvrtc/index.html#ixzz4ng9ZuEzf 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook

*/