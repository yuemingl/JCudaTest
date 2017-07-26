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
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
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

//javac -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcMultiGPU.java
//java -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcMultiGPU

public class JNvrtcMultiGPUVectorDotProduct {
    private static String programSourceCode = 
    "extern \"C\"" + "\n" +
    "#define THREADS_PER_BLOCK 256" + "\n" +
    "__global__ void dot(int n, float *a, float *b, float *sum)" + "\n" +
    "{" + "\n" +
    "    __shared__ float temp[THREADS_PER_BLOCK];" + "\n" +
    "    int index = threadIdx.x + blockIdx.x * blockDim.x;" + "\n" +
    "    if (index < n)" + "\n" +
    "    {" + "\n" +
    "        temp[threadIdx.x] = a[index] * b[index];" + "\n" +
    "    }" + "\n" +
    "    __syncthreads();" + "\n" +
    "    if (threadIdx.x == 0)" + "\n" +
    "    {" + "\n" +
    "        float localSum = 0;" + "\n" +
    "        for (int i = 0; i < THREADS_PER_BLOCK; i++)" + "\n" +
    "        {" + "\n" +
    "        	localSum += temp[i];" + "\n" +
    "        }" + "\n" +
    "        atomicAdd(sum, localSum);" + "\n" +
    "    }" + "\n" +
    "}" + "\n";

	static float[] aryGPUResults; // The length should be equal to number of GPUs

	public static class Runner implements Runnable {
		int ordinal; //GPU ordinal
		String ptx;  //parallel thread execution (PTX) instruction
		float[] v1;  // host input vector1
		float[] v2;  // host input vector2

		public Runner(int ordinal, String ptx, float[] v1, float[] v2) {
			this.ordinal = ordinal;
			this.ptx = ptx;
			this.v1 = v1;
			this.v2 = v2;
		}

		public void run() {
			// create a context for the device.
			CUdevice device = new CUdevice();
			cuDeviceGet(device, ordinal);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);

			// Create a CUDA module from the PTX code
			CUmodule module = new CUmodule();
			cuModuleLoadData(module, ptx);

			// Obtain the function pointer to the "dot" function from the module
			CUfunction function = new CUfunction();
			cuModuleGetFunction(function, module, "dot");

			int numElements = v1.length;

			// Allocate the device input data, and copy the
			// host input data to the device
			CUdeviceptr deviceInputA = new CUdeviceptr();
			cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT);
			cuMemcpyHtoD(deviceInputA, Pointer.to(v1), numElements * Sizeof.FLOAT);
			CUdeviceptr deviceInputB = new CUdeviceptr();
			cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT);
			cuMemcpyHtoD(deviceInputB, Pointer.to(v2), numElements * Sizeof.FLOAT);

			// Allocate device output memory
			CUdeviceptr deviceOutput = new CUdeviceptr();
			cuMemAlloc(deviceOutput, Sizeof.FLOAT);
			cuMemcpyHtoD(deviceOutput, Pointer.to(new float[] { 0.0f }), Sizeof.FLOAT);

			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			Pointer kernelParameters = Pointer.to(Pointer.to(new int[] { numElements }), Pointer.to(deviceInputA),
					Pointer.to(deviceInputB), Pointer.to(deviceOutput));

			// Call the kernel function, which was obtained from the
			// module that was compiled at runtime
			int blockSizeX = 256;
			int gridSizeX = (numElements + blockSizeX - 1) / blockSizeX;
			cuLaunchKernel(function, gridSizeX, 1, 1, // Grid dimension
					blockSizeX, 1, 1, // Block dimension
					0, null, // Shared memory size and stream
					kernelParameters, null // Kernel- and extra parameters
			);
			cuCtxSynchronize();

			// Allocate host output memory and copy the device output
			// to the host.
			float hostOutput[] = new float[1];
			cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, Sizeof.FLOAT);

			// Return the result
			System.out.println("Result from GPU " + this.ordinal + ": " + hostOutput[0]);
			aryGPUResults[this.ordinal] = hostOutput[0];

			// Clean up.
			cuMemFree(deviceInputA);
			cuMemFree(deviceInputB);
			cuMemFree(deviceOutput);

		}
	}

	public static float dotProduct(float[] v1, float[] v2) {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);
		JNvrtc.setExceptionsEnabled(true);
		
		// Initialize the driver
		cuInit(0);
		
		int[] aryCount = new int[1];
		cuDeviceGetCount(aryCount);
		int nGPU = aryCount[0];
		System.out.println("Total number of GPUs: " + nGPU);

		// initialize the result array for each GPU
		aryGPUResults = new float[nGPU];

		// Use the NVRTC to create a program by compiling the source code
		nvrtcProgram program = new nvrtcProgram();
		nvrtcCreateProgram(program, programSourceCode, null, 0, null, null);
		nvrtcCompileProgram(program, 0, null);

		// Print the compilation log (for the case there are any warnings)
		String programLog[] = new String[1];
		nvrtcGetProgramLog(program, programLog);
		System.out.println("Program compilation log:\n" + programLog[0]);

		// Obtain the PTX ("CUDA Assembler") code of the compiled program
		String[] ptx = new String[1];
		nvrtcGetPTX(program, ptx);
		System.out.println(ptx[0]);
		nvrtcDestroyProgram(program);

		// Start a thread for each GPU
		Thread[] ts = new Thread[nGPU];
		for (int i = 0; i < nGPU; i++) {
			ts[i] = new Thread(new Runner(i, ptx[0], 
					getSubVector(v1, i, nGPU), 
					getSubVector(v2, i, nGPU)));
			ts[i].start();
		}
		
		try {
			for (int i = 0; i < nGPU; i++) {
				ts[i].join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		float ret = 0.0f;
		for (int i = 0; i < nGPU; i++)
			ret += aryGPUResults[i];
		return ret;
	}

	public static float[] getSubVector(float[] vector, int nGPUOrdinal, int nTotalGPU) {
		int subLen = vector.length / nTotalGPU;
		int subRem = vector.length % nTotalGPU;
		int len = subLen;
		if (nGPUOrdinal == nTotalGPU - 1)
			len += subRem;
		float[] ret = new float[len];
		System.arraycopy(vector, subLen * nGPUOrdinal, ret, 0, ret.length);
		return ret;
	}

	public static void main(String[] args) throws InterruptedException {
		int N = 256*256;
		float[] v1 = new float[N];
		float[] v2 = new float[N];
		for (int i = 0; i < N; i++) {
			v1[i] = i;
			v2[i] = 1;
		}
		System.out.println(programSourceCode);
		System.out.println("The dot product (v1, v2) on GPU = " + dotProduct(v1, v2));
		float sum = 0.0f;
		for(int i=0; i<N; i++) {
			sum += v1[i]*v2[i];
		}
		System.out.println("The dot product (v1, v2) on CPU = " + sum);
	}
}

//Run:
//java -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcMultiGPUVectorDotProduct

//----Output-----

//extern "C"
//#define THREADS_PER_BLOCK 256
//__global__ void dot(int n, float *a, float *b, float *sum)
//{
//    __shared__ float temp[THREADS_PER_BLOCK];
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    if (index < n)
//    {
//        temp[threadIdx.x] = a[index] * b[index];
//    }
//    __syncthreads();
//    if (threadIdx.x == 0)
//    {
//        float localSum = 0;
//        for (int i = 0; i < THREADS_PER_BLOCK; i++)
//        {
//        	localSum += temp[i];
//        }
//        atomicAdd(sum, localSum);
//    }
//}
//Total number of GPUs: 8
//Program compilation log:
//
////
//// Generated by NVIDIA NVVM Compiler
////
//// Compiler Build ID: CL-21554848
//// Cuda compilation tools, release 8.0, V8.0.61
//// Based on LLVM 3.4svn
////
//
//.version 5.0
//.target sm_20
//.address_size 64
//
//	// .globl	dot
//// dot$temp has been demoted
//
//.visible .entry dot(
//	.param .u32 dot_param_0,
//	.param .u64 dot_param_1,
//	.param .u64 dot_param_2,
//	.param .u64 dot_param_3
//)
//{
//	.reg .pred 	%p<4>;
//	.reg .f32 	%f<40>;
//	.reg .b32 	%r<10>;
//	.reg .b64 	%rd<17>;
//	// demoted variable
//	.shared .align 4 .b8 dot$temp[1024];
//
//	ld.param.u32 	%r5, [dot_param_0];
//	ld.param.u64 	%rd4, [dot_param_1];
//	ld.param.u64 	%rd5, [dot_param_2];
//	ld.param.u64 	%rd6, [dot_param_3];
//	mov.u32 	%r6, %ntid.x;
//	mov.u32 	%r7, %ctaid.x;
//	mov.u32 	%r1, %tid.x;
//	mad.lo.s32 	%r2, %r6, %r7, %r1;
//	setp.ge.s32	%p1, %r2, %r5;
//	@%p1 bra 	BB0_2;
//
//	cvta.to.global.u64 	%rd7, %rd4;
//	mul.wide.s32 	%rd8, %r2, 4;
//	add.s64 	%rd9, %rd7, %rd8;
//	cvta.to.global.u64 	%rd10, %rd5;
//	add.s64 	%rd11, %rd10, %rd8;
//	ld.global.f32 	%f3, [%rd11];
//	ld.global.f32 	%f4, [%rd9];
//	mul.f32 	%f5, %f4, %f3;
//	mul.wide.u32 	%rd12, %r1, 4;
//	mov.u64 	%rd13, dot$temp;
//	add.s64 	%rd14, %rd13, %rd12;
//	st.shared.f32 	[%rd14], %f5;
//
//BB0_2:
//	cvta.to.global.u64 	%rd1, %rd6;
//	bar.sync 	0;
//	mov.f32 	%f39, 0f00000000;
//	mov.u32 	%r9, -256;
//	mov.u64 	%rd16, dot$temp;
//	setp.ne.s32	%p2, %r1, 0;
//	@%p2 bra 	BB0_5;
//
//BB0_3:
//	ld.shared.f32 	%f7, [%rd16];
//	add.f32 	%f8, %f39, %f7;
//	ld.shared.f32 	%f9, [%rd16+4];
//	add.f32 	%f10, %f8, %f9;
//	ld.shared.f32 	%f11, [%rd16+8];
//	add.f32 	%f12, %f10, %f11;
//	ld.shared.f32 	%f13, [%rd16+12];
//	add.f32 	%f14, %f12, %f13;
//	ld.shared.f32 	%f15, [%rd16+16];
//	add.f32 	%f16, %f14, %f15;
//	ld.shared.f32 	%f17, [%rd16+20];
//	add.f32 	%f18, %f16, %f17;
//	ld.shared.f32 	%f19, [%rd16+24];
//	add.f32 	%f20, %f18, %f19;
//	ld.shared.f32 	%f21, [%rd16+28];
//	add.f32 	%f22, %f20, %f21;
//	ld.shared.f32 	%f23, [%rd16+32];
//	add.f32 	%f24, %f22, %f23;
//	ld.shared.f32 	%f25, [%rd16+36];
//	add.f32 	%f26, %f24, %f25;
//	ld.shared.f32 	%f27, [%rd16+40];
//	add.f32 	%f28, %f26, %f27;
//	ld.shared.f32 	%f29, [%rd16+44];
//	add.f32 	%f30, %f28, %f29;
//	ld.shared.f32 	%f31, [%rd16+48];
//	add.f32 	%f32, %f30, %f31;
//	ld.shared.f32 	%f33, [%rd16+52];
//	add.f32 	%f34, %f32, %f33;
//	ld.shared.f32 	%f35, [%rd16+56];
//	add.f32 	%f36, %f34, %f35;
//	ld.shared.f32 	%f37, [%rd16+60];
//	add.f32 	%f39, %f36, %f37;
//	add.s64 	%rd16, %rd16, 64;
//	add.s32 	%r9, %r9, 16;
//	setp.ne.s32	%p3, %r9, 0;
//	@%p3 bra 	BB0_3;
//
//	atom.global.add.f32 	%f38, [%rd1], %f39;
//
//BB0_5:
//	ret;
//}
//
//
//Result from GPU 7: 5.03312384E8
//Result from GPU 1: 1.006592E8
//Result from GPU 4: 3.01985792E8
//Result from GPU 2: 1.67768064E8
//Result from GPU 0: 3.3550336E7
//Result from GPU 3: 2.34876928E8
//Result from GPU 6: 4.3620352E8
//Result from GPU 5: 3.69094656E8
//The dot product (v1, v2) on GPU = 2.14745088E9
//The dot product (v1, v2) on CPU = 2.14742118E9