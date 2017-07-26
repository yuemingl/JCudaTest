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

//javac -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcMultiGPUTemplate.java
//java -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JNvrtcMultiGPUTemplate

public class JNvrtcMultiGPUTemplate
{
    private static String programSourceCode = 
        "extern \"C\"" + "\n" +
        "__global__ void add(int n, float *a, float *b, float *sum)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        sum[0] += a[i] + b[i];" + "\n" +
        "    }" + "\n" +
        "}" + "\n";

    public static class Runner implements Runnable {
    	int ordinal;
    	String ptx;
    	
    	public Runner(int ordinal, String ptx) {
    		this.ordinal = ordinal;
    		this.ptx = ptx;
    	}
    	
        public void run() {
        	//create a context for the device.
            CUdevice device = new CUdevice();
            cuDeviceGet(device, ordinal);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            // Create a CUDA module from the PTX code
            CUmodule module = new CUmodule();
            cuModuleLoadData(module, ptx);

            // Obtain the function pointer to the "add" function from the module
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, "add");

            // Continue with some basic setup for the vector addition itself:

            // Allocate and fill the host input data
            int numElements = 8;
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
            cuMemAlloc(deviceOutput, Sizeof.FLOAT);
            cuMemcpyHtoD(deviceOutput, Pointer.to(new float[]{0.0f}),
                Sizeof.FLOAT);

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
            float hostOutput[] = new float[1];
            cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
                Sizeof.FLOAT);

            // Verify the result
            System.out.println("Result from GPU "+this.ordinal+": "+hostOutput[0]);

            // Clean up.
            cuMemFree(deviceInputA);
            cuMemFree(deviceInputB);
            cuMemFree(deviceOutput);

        }
    }
    public static void main(String[] args) throws InterruptedException
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        // Initialize the driver
        cuInit(0);
        int[] count = new int[1];
        cuDeviceGetCount(count);
        System.out.println("Total GPU: "+count[0]);
        
        
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
        System.out.println(ptx[0]);
        nvrtcDestroyProgram(program);
       
        int N = count[0];
        Thread[] ts = new Thread[N];
        for(int i=0; i<N; i++) {
        	ts[i] = new Thread(new Runner(i, ptx[0]));
        	ts[i].start();
        }
        for(int i=0; i<N; i++) {
        	ts[i].join();
        }
    }
}

////
////Generated by NVIDIA NVVM Compiler
////
////Compiler Build ID: CL-21554848
////Cuda compilation tools, release 8.0, V8.0.61
////Based on LLVM 3.4svn
////
//
//.version 5.0
//.target sm_20
//.address_size 64
//
//	// .globl	add
//
//.visible .entry add(
//	.param .u32 add_param_0,
//	.param .u64 add_param_1,
//	.param .u64 add_param_2,
//	.param .u64 add_param_3
//)
//{
//	.reg .pred 	%p<2>;
//	.reg .f32 	%f<6>;
//	.reg .b32 	%r<6>;
//	.reg .b64 	%rd<10>;
//
//
//	ld.param.u32 	%r2, [add_param_0];
//	ld.param.u64 	%rd1, [add_param_1];
//	ld.param.u64 	%rd2, [add_param_2];
//	ld.param.u64 	%rd3, [add_param_3];
//	mov.u32 	%r3, %ntid.x;
//	mov.u32 	%r4, %ctaid.x;
//	mov.u32 	%r5, %tid.x;
//	mad.lo.s32 	%r1, %r3, %r4, %r5;
//	setp.ge.s32	%p1, %r1, %r2;
//	@%p1 bra 	BB0_2;
//
//	cvta.to.global.u64 	%rd4, %rd3;
//	cvta.to.global.u64 	%rd5, %rd1;
//	mul.wide.s32 	%rd6, %r1, 4;
//	add.s64 	%rd7, %rd5, %rd6;
//	cvta.to.global.u64 	%rd8, %rd2;
//	add.s64 	%rd9, %rd8, %rd6;
//	ld.global.f32 	%f1, [%rd9];
//	ld.global.f32 	%f2, [%rd7];
//	add.f32 	%f3, %f2, %f1;
//	ldu.global.f32 	%f4, [%rd4];
//	add.f32 	%f5, %f4, %f3;
//	st.global.f32 	[%rd4], %f5;
//
//BB0_2:
//	ret;
//}
