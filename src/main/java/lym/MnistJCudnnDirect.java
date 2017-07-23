package lym;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSgemv;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcudnn.JCudnn.CUDNN_VERSION;
import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnFindConvolutionForwardAlgorithm;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionNdForwardOutputDim;
import static jcuda.jcudnn.JCudnn.cudnnGetErrorString;
import static jcuda.jcudnn.JCudnn.cudnnGetPoolingNdForwardOutputDim;
import static jcuda.jcudnn.JCudnn.cudnnGetVersion;
import static jcuda.jcudnn.JCudnn.cudnnLRNCrossChannelForward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolutionNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilterNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPoolingNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxForward;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnAddMode.CUDNN_ADD_SAME_C;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
import static jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdAlgo;
import jcuda.jcudnn.cudnnConvolutionFwdAlgoPerf;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

/**
 * A direct port of the "mnistCUDNN" sample.<br> 
 * <br>
 * This sample expects the data files that are part of the 
 * mnistCUDNN sample to be present in a "data/" subdirectory.
 */
public class MnistJCudnnDirect
{
    public static final int IMAGE_H  = 28;
    public static final int IMAGE_W  = 28;

    public static final String first_image = "one_28x28.pgm";
    public static final String second_image = "three_28x28.pgm";
    public static final String third_image = "five_28x28.pgm";

    public static final String conv1_bin = "conv1.bin";
    public static final String conv1_bias_bin = "conv1.bias.bin";
    public static final String conv2_bin = "conv2.bin";
    public static final String conv2_bias_bin = "conv2.bias.bin";
    public static final String ip1_bin = "ip1.bin";
    public static final String ip1_bias_bin = "ip1.bias.bin";
    public static final String ip2_bin = "ip2.bin";
    public static final String ip2_bias_bin = "ip2.bias.bin";

    static void readBinaryFile(String fname, int size, float data_h[]) throws IOException
    {
        FileInputStream fis = new FileInputStream(new File(fname));
        byte data[] = readFully(fis);
        ByteBuffer bb = ByteBuffer.wrap(data);
        FloatBuffer input = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        FloatBuffer output = FloatBuffer.wrap(data_h);
        output.put(input);
    }

    private static byte[] readFully(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[1024];
        while (true)
        {
            int n = inputStream.read(buffer);
            if (n < 0)
            {
                break;
            }
            baos.write(buffer, 0, n);
        }
        byte data[] = baos.toByteArray();
        return data;
    }


    static void readAllocMemcpy(String fname, int size, Pointer data_h[], Pointer data_d[])
    {
        float data_h_Array[] = new float[size];
        try
        {
            readBinaryFile(fname, size, data_h_Array);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        int size_b = size*Sizeof.FLOAT;
        Pointer data_d_Pointer = new Pointer();
        cudaMalloc(data_d_Pointer, size_b);
        cudaMemcpy(data_d_Pointer, Pointer.to(data_h_Array), size_b, cudaMemcpyHostToDevice);
        data_h[0] = Pointer.to(data_h_Array);
        data_d[0] = data_d_Pointer;
    }

    private static byte[] readBinaryPortableGraymap8bitData(
        InputStream inputStream) throws IOException
    {
        DataInputStream dis = new DataInputStream(inputStream);  
        String line = null;
        boolean firstLine = true;
        Integer width = null;
        Integer height = null;
        Integer maxBrightness = null;
        while (true)
        {
            // The DataInputStream#readLine is deprecated,
            // but for ASCII input, it is safe to use it
            line = dis.readLine();
            if (line == null)
            {
                break;
            }
            line = line.trim();
            if (line.startsWith("#"))
            {
                continue;
            }
            if (firstLine)
            {
                firstLine = false;
                if (!line.equals("P5"))
                {
                    throw new IOException(
                        "Data is not a binary portable " + 
                            "graymap (P5), but "+line);
                }
                else
                {
                    continue;
                }
            }
            if (width == null)
            {
                String tokens[] = line.split(" ");
                if (tokens.length < 2)
                {
                    throw new IOException("Expected dimensions, found "+line);
                }
                width = parseInt(tokens[0]);
                height = parseInt(tokens[1]);
            }
            else if (maxBrightness == null)
            {
                maxBrightness = parseInt(line);
                if (maxBrightness > 255)
                {
                    throw new IOException(
                        "Only 8 bit values supported. " +
                            "Maximum value is "+maxBrightness);
                }
                break;
            }
        }
        byte data[] = readFully(inputStream);
        return data;
    }

    private static Integer parseInt(String s) throws IOException
    {
        try
        {
            return Integer.parseInt(s);
        }
        catch (NumberFormatException e)
        {
            throw new IOException(e);
        }
    }

    static void readImage(String fname, float imgData_h[])
    {
        try
        {
            InputStream is = new FileInputStream(new File(fname));
            byte data[] = readBinaryPortableGraymap8bitData(is);
            
            for (int i = 0; i < IMAGE_H; i++)
            {
                for (int j = 0; j < IMAGE_W; j++)
                {   
                    int idx = IMAGE_W*i + j;
                    imgData_h[idx] = (((int) data[idx]) & 0xff) / 255.0f;
                }
            } 
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    static void printDeviceVector(int size, Pointer vec_d)
    {
        float vec[] = new float[size];
        cudaDeviceSynchronize();
        cudaMemcpy(Pointer.to(vec), vec_d, size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++)
        {
            System.out.print(vec[i] + " ");
        }
        System.out.println();
    }

    static class Layer_t
    {
        int inputs;
        int outputs;
        // linear dimension (i.e. size is kernel_dim * kernel_dim)
        int kernel_dim;
        Pointer data_h;
        Pointer data_d;
        Pointer bias_h;
        Pointer bias_d;

        Layer_t(int _inputs, int _outputs, int _kernel_dim, String fname_weights, String fname_bias)
        {
            this.inputs = _inputs;
            this.outputs = _outputs;
            this.kernel_dim = _kernel_dim;

            String weights_path = "data/"+fname_weights;
            String bias_path = "data/"+fname_bias;

            Pointer h[] = { null };
            Pointer d[] = { null };
            readAllocInit(weights_path, inputs * outputs * kernel_dim * kernel_dim, h, d);
            data_h = h[0];
            data_d = d[0];

            readAllocInit(bias_path, outputs, h, d);
            bias_h = h[0];
            bias_d = d[0];
        }
        void destroyLayer()
        {
            cudaFree(data_d);
            cudaFree(bias_d);
        }

        void readAllocInit(String fname, int size, Pointer data_h[], Pointer data_d[])
        {
            readAllocMemcpy(fname, size, data_h, data_d);
        }
    };


    static void setTensorDesc(cudnnTensorDescriptor tensorDesc, 
        int tensorFormat, // cudnnTensorFormat_t
        int dataType, // cudnnDataType_t
        int n,
        int c,
        int h,
        int w)
    {
        cudnnSetTensor4dDescriptor(
            tensorDesc, tensorFormat, dataType, n, c, h, w);
    }


    static class network_t
    {
        int convAlgorithm;
        int dataType; // cudnnDataType_t
        int tensorFormat; // cudnnTensorFormat_t
        cudnnHandle cudnnHandle;
        cudnnTensorDescriptor srcTensorDesc, dstTensorDesc, biasTensorDesc;
        cudnnFilterDescriptor filterDesc;
        cudnnConvolutionDescriptor convDesc;
        cudnnPoolingDescriptor poolingDesc;
        cudnnLRNDescriptor   normDesc;
        cublasHandle cublasHandle;

        void createHandles()
        {
            cudnnHandle = new cudnnHandle();
            srcTensorDesc = new cudnnTensorDescriptor();
            dstTensorDesc = new cudnnTensorDescriptor();
            biasTensorDesc = new cudnnTensorDescriptor();
            filterDesc = new cudnnFilterDescriptor();
            convDesc = new cudnnConvolutionDescriptor();
            poolingDesc = new cudnnPoolingDescriptor();
            normDesc = new cudnnLRNDescriptor();

            cudnnCreate(cudnnHandle);
            cudnnCreateTensorDescriptor(srcTensorDesc);
            cudnnCreateTensorDescriptor(dstTensorDesc);
            cudnnCreateTensorDescriptor(biasTensorDesc);
            cudnnCreateFilterDescriptor(filterDesc);
            cudnnCreateConvolutionDescriptor(convDesc);
            cudnnCreatePoolingDescriptor(poolingDesc);
            cudnnCreateLRNDescriptor(normDesc);

            cublasHandle = new cublasHandle();
            cublasCreate(cublasHandle);
        }
        void destroyHandles()
        {
            cudnnDestroyLRNDescriptor(normDesc);
            cudnnDestroyPoolingDescriptor(poolingDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            cudnnDestroyFilterDescriptor(filterDesc);
            cudnnDestroyTensorDescriptor(srcTensorDesc);
            cudnnDestroyTensorDescriptor(dstTensorDesc);
            cudnnDestroyTensorDescriptor(biasTensorDesc);
            cudnnDestroy(cudnnHandle);

            cublasDestroy(cublasHandle);
        }
        network_t()
        {
            convAlgorithm = -1;
            dataType = CUDNN_DATA_FLOAT; 
            tensorFormat = CUDNN_TENSOR_NCHW;
            createHandles();    
        };
        void resize(int size, Pointer data)
        {
            if (data != null)
            {
                cudaFree(data);
            }
            cudaMalloc(data, size*Sizeof.FLOAT);
        }
        void setConvolutionAlgorithm(int algo)
        {
            convAlgorithm = (int) algo;
        }
        void addBias( cudnnTensorDescriptor dstTensorDesc, Layer_t layer, int c, Pointer data)
        {
            setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);
            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 1.0f });
            cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
                alpha, biasTensorDesc,
                layer.bias_d,
                beta,
                dstTensorDesc,
                data);
        }

        void fullyConnectedForward(Layer_t ip,
            int n[], int c[], int h[], int w[],
            Pointer srcData, Pointer dstData)
        {
            if (n[0] != 1)
            {
                System.out.println("Not Implemented");
                System.exit(-1);
            }
            int dim_x = c[0]*h[0]*w[0];
            int dim_y = ip.outputs;
            resize(dim_y, dstData);

            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 1.0f });

            // place bias into dstData
            cudaMemcpy(dstData, ip.bias_d, dim_y*Sizeof.FLOAT, cudaMemcpyDeviceToDevice);

            gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta, dstData);

            h[0] = 1; w[0] = 1; c[0] = dim_y;
        }

        static void gemv(cublasHandle cublasHandle, int m, int n, Pointer alpha, 
            Pointer A, Pointer x,
            Pointer beta, Pointer y)
        {
            cublasSgemv(cublasHandle, CUBLAS_OP_T,
                m, n,
                alpha,
                A, m,
                x, 1,
                beta,
                y, 1);    
        };

        void convoluteForward(Layer_t conv,
            int n[], int c[], int h[], int w[],
            Pointer srcData, Pointer dstData)
        {
            int algo = 0; // cudnnConvolutionFwdAlgo_t

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);

            int tensorDims = 4;
            int tensorOuputDimA[] = {n[0],c[0],h[0],w[0]};
            int filterDimA[] = {
                conv.outputs, conv.inputs, 
                conv.kernel_dim, conv.kernel_dim};

            cudnnSetFilterNdDescriptor(filterDesc,
                dataType,
                tensorDims,
                filterDimA);

            int convDims = 2;
            int padA[] = {0,0};
            int filterStrideA[] = {1,1};
            int upscaleA[] = {1,1};
            cudnnSetConvolutionNdDescriptor(convDesc,
                convDims,
                padA,
                filterStrideA,
                upscaleA,
                CUDNN_CROSS_CORRELATION);

            // find dimension of convolution output
            cudnnGetConvolutionNdForwardOutputDim(convDesc,
                srcTensorDesc,
                filterDesc,
                tensorDims,
                tensorOuputDimA);
            n[0] = tensorOuputDimA[0]; 
            c[0] = tensorOuputDimA[1];
            h[0] = tensorOuputDimA[2]; 
            w[0] = tensorOuputDimA[3];

            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);

            if (convAlgorithm < 0)
            {
                int algoArray[] = { -1 };
                // Choose the best according to the preference
                System.out.println("Testing cudnnGetConvolutionForwardAlgorithm ...");
                cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                    srcTensorDesc,
                    filterDesc,
                    convDesc,
                    dstTensorDesc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0,
                    algoArray
                    );
                algo = algoArray[0];

                System.out.println("Fastest algorithm is Algo " + algo);
                convAlgorithm = algo;

                // New way of finding the fastest config
                // Setup for findFastest call
                System.out.println("Testing cudnnFindConvolutionForwardAlgorithm ...");
                int requestedAlgoCount = 5; 
                int returnedAlgoCount[] = new int[1];
                cudnnConvolutionFwdAlgoPerf results[] = 
                    new cudnnConvolutionFwdAlgoPerf[requestedAlgoCount];
                cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
                    srcTensorDesc,
                    filterDesc,
                    convDesc,
                    dstTensorDesc,
                    requestedAlgoCount,
                    returnedAlgoCount,
                    results
                    );
                for(int algoIndex = 0; algoIndex < returnedAlgoCount[0]; ++algoIndex){
                    System.out.printf("^^^^ %s for Algo %d (%s): %f time requiring %d memory\n", 
                        cudnnGetErrorString(results[algoIndex].status), 
                        results[algoIndex].algo,
                        cudnnConvolutionFwdAlgo.stringFor(results[algoIndex].algo), 
                        results[algoIndex].time, results[algoIndex].memory);
                }
            }
            else
            {
                algo = convAlgorithm;
                if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
                {
                    System.out.println("Using FFT for convolution");
                }
            }

            resize(n[0]*c[0]*h[0]*w[0], dstData);
            long sizeInBytesArray[] = { 0 };
            Pointer workSpace= new Pointer();
            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcTensorDesc,
                filterDesc,
                convDesc,
                dstTensorDesc,
                algo,
                sizeInBytesArray);
            long sizeInBytes = sizeInBytesArray[0];
            if (sizeInBytes!=0)
            {
                cudaMalloc(workSpace,sizeInBytes);
            }
            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 0.0f });
            cudnnConvolutionForward(cudnnHandle,
                alpha,
                srcTensorDesc,
                srcData,
                filterDesc,
                conv.data_d,
                convDesc,
                algo,
                workSpace,
                sizeInBytes,
                beta,
                dstTensorDesc,
                dstData);
            addBias(dstTensorDesc, conv, c[0], dstData);
            if (sizeInBytes!=0)
            {
                cudaFree(workSpace);
            }
        }


        void poolForward( int n[], int c[], int h[], int w[],
            Pointer srcData, Pointer dstData)
        {
            int poolDims = 2;
            int windowDimA[] = {2,2};
            int paddingA[] = {0,0};
            int strideA[] = {2,2};
            cudnnSetPoolingNdDescriptor(poolingDesc,
                CUDNN_POOLING_MAX,
                poolDims,
                windowDimA,
                paddingA,
                strideA);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);        

            int tensorDims = 4;
            int tensorOuputDimA[] = {n[0],c[0],h[0],w[0]};
            cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                srcTensorDesc,
                tensorDims,
                tensorOuputDimA);
            n[0] = tensorOuputDimA[0]; 
            c[0] = tensorOuputDimA[1];
            h[0] = tensorOuputDimA[2]; 
            w[0] = tensorOuputDimA[3];

            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);  

            resize(n[0]*c[0]*h[0]*w[0], dstData);
            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 0.0f });
            cudnnPoolingForward(cudnnHandle,
                poolingDesc,
                alpha,
                srcTensorDesc,
                srcData,
                beta,
                dstTensorDesc,
                dstData);
        }

        void softmaxForward(int n[], int c[], int h[], int w[], Pointer srcData, Pointer dstData)
        {
            resize(n[0]*c[0]*h[0]*w[0], dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);

            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 0.0f });
            cudnnSoftmaxForward(cudnnHandle,
                CUDNN_SOFTMAX_ACCURATE ,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                alpha,
                srcTensorDesc,
                srcData,
                beta,
                dstTensorDesc,
                dstData);
        }

        void lrnForward(int n[], int c[], int h[], int w[], Pointer srcData, Pointer dstData)
        {
            int lrnN = 5;
            double lrnAlpha, lrnBeta, lrnK;
            lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
            cudnnSetLRNDescriptor(normDesc,
                lrnN,
                lrnAlpha,
                lrnBeta,
                lrnK);

            resize(n[0]*c[0]*h[0]*w[0], dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);

            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 0.0f });
            cudnnLRNCrossChannelForward(cudnnHandle,
                normDesc,
                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                alpha,
                srcTensorDesc,
                srcData,
                beta,
                dstTensorDesc,
                dstData);
        }


        void activationForward(int n[], int c[], int h[], int w[], Pointer srcData, Pointer dstData)
        {
            resize(n[0]*c[0]*h[0]*w[0], dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n[0], c[0], h[0], w[0]);

            Pointer alpha = Pointer.to(new float[] { 1.0f });
            Pointer beta = Pointer.to(new float[] { 0.0f });
            cudnnActivationForward(cudnnHandle,
                CUDNN_ACTIVATION_RELU,
                alpha,
                srcTensorDesc,
                srcData,
                beta,
                dstTensorDesc,
                dstData);    
        }

        int classify_example(String fname, 
            Layer_t conv1,
            Layer_t conv2,
            Layer_t ip1,
            Layer_t ip2)
        {
            int n[] = { 0 };
            int c[] = { 0 };
            int h[] = { 0 };
            int w[] = { 0 };
            Pointer srcData = new Pointer();
            Pointer dstData = new Pointer();
            float imgData_h[] = new float[IMAGE_H*IMAGE_W];

            readImage(fname, imgData_h);

            System.out.println("Performing forward propagation ...");

            cudaMalloc(srcData, IMAGE_H*IMAGE_W*Sizeof.FLOAT);
            cudaMemcpy(srcData, Pointer.to(imgData_h),
                IMAGE_H*IMAGE_W*Sizeof.FLOAT,
                cudaMemcpyHostToDevice);

            n[0] = c[0] = 1; 
            h[0] = IMAGE_H; 
            w[0] = IMAGE_W;
            convoluteForward(conv1, n, c, h, w, srcData, dstData);
            poolForward(n, c, h, w, dstData, srcData);

            convoluteForward(conv2, n, c, h, w, srcData, dstData);
            poolForward(n, c, h, w, dstData, srcData);

            fullyConnectedForward(ip1, n, c, h, w, srcData, dstData);
            activationForward(n, c, h, w, dstData, srcData);
            lrnForward(n, c, h, w, srcData, dstData);

            fullyConnectedForward(ip2, n, c, h, w, dstData, srcData);
            softmaxForward(n, c, h, w, srcData, dstData);

            int max_digits = 10;
            float result[] = new float[max_digits];
            cudaMemcpy(Pointer.to(result), dstData, max_digits*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            int id = 0;
            for (int i = 1; i < max_digits; i++)
            {
                if (result[id] < result[i]) id = i;
            }

            System.out.println("Resulting weights from Softmax:");
            printDeviceVector(n[0]*c[0]*h[0]*w[0], dstData);

            cudaFree(srcData);
            cudaFree(dstData);
            return id;
        }
    }

    public static void main(String args[])
    {   
        JCuda.setExceptionsEnabled(true);
        JCudnn.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);

        int version = (int)cudnnGetVersion();
        System.out.printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d\n", version, CUDNN_VERSION);
        System.out.println("\nTesting single precision\n");
        network_t mnist = new network_t();
        Layer_t conv1 = new Layer_t(1,20,5,conv1_bin,conv1_bias_bin);
        Layer_t conv2 = new Layer_t(20,50,5,conv2_bin,conv2_bias_bin);
        Layer_t   ip1 = new Layer_t(800,500,1,ip1_bin,ip1_bias_bin);
        Layer_t   ip2 = new Layer_t(500,10,1,ip2_bin,ip2_bias_bin);
        int i1 = mnist.classify_example("data/"+first_image, conv1, conv2, ip1, ip2);
        int i2 = mnist.classify_example("data/"+second_image, conv1, conv2, ip1, ip2);

        mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
        int i3 = mnist.classify_example("data/"+third_image, conv1, conv2, ip1, ip2);

        System.out.println("\nResult of classification: " + i1 + " " + i2 + " " + i3);
        if (i1 != 1 || i2 != 3 || i3 != 5)
        {
            System.out.println("\nTest failed!\n");
        }
        else
        {
            System.out.println("\nTest passed!\n");
        }
    }

}