package lym;

import static jcuda.jcudnn.JCudnn.CUDNN_VERSION;
import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetVersion;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

//javac -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" lym/JCudnnBasic.java
//java -cp ".:jcuda-0.8.0.jar:jcuda-natives-0.8.0-linux-x86_64.jar:jcublas-0.8.0.jar:jcublas-natives-0.8.0-linux-x86_64.jar:jcudnn-0.8.0.jar:jcudnn-natives-0.8.0-linux-x86_64.jar" JCudnnBasic
public class JCudnnBasic {
	public static void main(String args[]) {
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);

		int version = (int) cudnnGetVersion();
		System.out.printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d\n", version, CUDNN_VERSION);

		cudnnHandle cudnnHandle = new cudnnHandle();
		cudnnCreate(cudnnHandle);

		cudnnTensorDescriptor desc1 = new cudnnTensorDescriptor();
		cudnnTensorDescriptor desc2 = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(desc1);
		cudnnCreateTensorDescriptor(desc2);

		int n = 1; // the batch size
		int c = 1; // the number of feature maps
		int h = 28; // the height
		int w = 28; // the width
		cudnnSetTensor4dDescriptor(desc1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
		cudnnSetTensor4dDescriptor(desc2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

		Pointer alpha = Pointer.to(new float[] { 1.0f });
		Pointer beta = Pointer.to(new float[] { 1.0f });

		float imgHostData1[] = new float[h * w];
		for (int i = 0; i < h * w; i++)
			imgHostData1[i] = i;
		float imgHostData2[] = new float[h * w];
		for (int i = 0; i < h * w; i++)
			imgHostData2[i] = i * i;

		Pointer imgDeviceData1 = new Pointer();
		cudaMalloc(imgDeviceData1, h * w * Sizeof.FLOAT);
		Pointer imgDeviceData2 = new Pointer();
		cudaMalloc(imgDeviceData2, h * w * Sizeof.FLOAT);

		// copy host data to device
		cudaMemcpy(imgDeviceData1, Pointer.to(imgHostData1), h * w * Sizeof.FLOAT, cudaMemcpyHostToDevice);
		cudaMemcpy(imgDeviceData2, Pointer.to(imgHostData2), h * w * Sizeof.FLOAT, cudaMemcpyHostToDevice);

		// C = alpha * A + beta * C
		cudnnAddTensor(cudnnHandle, // CUDNN_ADD_SAME_C,
				alpha, desc1, imgDeviceData1, // A
				beta, desc2, imgDeviceData2); // C

		float result[] = new float[h * w];
		cudaMemcpy(Pointer.to(result), imgDeviceData2, h * w * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

		System.out.println(result[0]);
		System.out.println(result[1]);
		System.out.println(result[2]);

		cudnnDestroyTensorDescriptor(desc1);
		cudnnDestroyTensorDescriptor(desc2);
		cudnnDestroy(cudnnHandle);
	}

}
