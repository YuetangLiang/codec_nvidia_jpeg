#ifndef __NVJPEG_CODER_HPP__
#define __NVJPEG_CODER_HPP__

/**
 * ref:
 * https://docs.nvidia.com/cuda/nvjpeg/index.html
 */

#ifndef __D
#define __D(fmt, args...) printf("[%s:%3d] " fmt, __FUNCTION__, __LINE__, ## args)
#endif

#ifndef __E
#define __E(fmt, args...) fprintf(stderr, "" fmt, ## args)
#endif

#ifndef __I
#define __I(fmt, args...) printf("[%s:%3d] " fmt, __FUNCTION__, __LINE__, ## args)
#endif


//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <npp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>

#ifndef chkErrors
#define chkErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        __E("CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}
#endif

struct encode_params_t {
  std::string format;
  std::string subsampling;
  int quality;
  int huf;
  int dev;
  int input_w;
  int input_h;
};

class Nvjpeg_decoder
{
private:
    int dev;

public:
    unsigned char* m_pBuffer_gpu;
    cudaEvent_t startEvent, stopEvent;
    nvjpegHandle_t decoder_handle;
    nvjpegJpegState_t jpeg_state;
    size_t imageSize = 0;
    int nComponent = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t sampling_type;

    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_RGBI;

    static int dev_malloc(void **p, size_t s) {
        return (int)cudaMalloc(p, s);
    }

    static int dev_free(void *p) {
        return (int)cudaFree(p);
    }

    bool is_interleaved(nvjpegOutputFormat_t format) {
        return (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI);
    }

    Nvjpeg_decoder(int dev) {
        cudaDeviceProp props;
        chkErrors(cudaGetDeviceProperties(&props, dev));

        __D("Decoder Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
            dev, props.name, props.multiProcessorCount,
            props.maxThreadsPerMultiProcessor, props.major, props.minor,
            props.ECCEnabled ? "on" : "off");

        chkErrors(cudaSetDevice(dev));

        nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
        chkErrors(
            nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &decoder_handle));
        chkErrors(
            nvjpegJpegStateCreate(decoder_handle, &jpeg_state));

        chkErrors(
            cudaMalloc(&m_pBuffer_gpu, 3840 * 2160 * NVJPEG_MAX_COMPONENT));
        __D("nvjpeg Decoder support maximum resolution: 3840x2160 \n");

        chkErrors(
            cudaEventCreate(&startEvent, cudaEventBlockingSync));
        chkErrors(
            cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    };

    ~Nvjpeg_decoder() {
        chkErrors(
            cudaFree(m_pBuffer_gpu));
        chkErrors(
            nvjpegJpegStateDestroy(jpeg_state));
        chkErrors(
            nvjpegDestroy(decoder_handle));
        chkErrors(cudaEventDestroy(startEvent));
        chkErrors(cudaEventDestroy(stopEvent));
    };

    int decode_from_jpg(std::string image_path, unsigned char *img_dst) {
        FILE *streamFile = fopen(&image_path[0], "rb");
        if (!streamFile) {
            __E("Cannot open image: %s\n", &image_path[0]);
            return -1;
        }
        fseek(streamFile, 0, SEEK_END);
        auto bitstreamBytes = ftell(streamFile);
        fseek(streamFile, 0, SEEK_SET);

        std::vector<unsigned char> vBuffer(bitstreamBytes);
        auto bitstream = vBuffer.data();

        if (fread(bitstream, bitstreamBytes, 1, streamFile) != 1) {
            __E("Error JPEG file %s for %ld bytes \n", &image_path[0], bitstreamBytes);
            return -1;
        }

        return decode(bitstream, bitstreamBytes, img_dst);
    };

    std::string nvjpeg_status(nvjpegStatus_t code) {
        switch (code)
        {
        case NVJPEG_STATUS_SUCCESS:
            return "NVJPEG_STATUS_SUCCESS";
        case NVJPEG_STATUS_NOT_INITIALIZED:
            return "NVJPEG_STATUS_NOT_INITIALIZED";
        case NVJPEG_STATUS_INVALID_PARAMETER:
            return "NVJPEG_STATUS_INVALID_PARAMETER";
        case NVJPEG_STATUS_BAD_JPEG:
            return "NVJPEG_STATUS_BAD_JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
            return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE:
            return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
        case NVJPEG_STATUS_EXECUTION_FAILED:
            return "NVJPEG_STATUS_EXECUTION_FAILED";
        case NVJPEG_STATUS_ARCH_MISMATCH:
            return "NVJPEG_STATUS_ARCH_MISMATCH";
        case NVJPEG_STATUS_INTERNAL_ERROR:
            return "NVJPEG_STATUS_INTERNAL_ERROR";
        case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
            return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
        }

        return "Unknown NVJPEG_STATUS";
    };

    int decode(unsigned char* bitstream, size_t bitstreamBytes, unsigned char *img_dst)
    {
        chkErrors(
            nvjpegGetImageInfo(decoder_handle, bitstream, bitstreamBytes, &nComponent, &sampling_type, widths, heights));

        __D("Image is %d channels\n", nComponent);
        for (int i = 0; i < nComponent; i++) {
            __D("Channel #%d size: %dx%d \n", i, widths[i], heights[i]);
        }

        switch (sampling_type)
        {
        case NVJPEG_CSS_444:
            __D("YUV 4:4:4 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_440:
            __D("YUV 4:4:0 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_422:
            __D("YUV 4:2:2 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_420:
            __D("YUV 4:2:0 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_411:
            __D("YUV 4:1:1 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_410:
            __D("YUV 4:1:0 chroma sub-sampling type \n");
            break;
        case NVJPEG_CSS_GRAY:
            __D("Grayscale sub-sampling type \n");
            break;
        case NVJPEG_CSS_UNKNOWN:
            __E("Unknown chroma sub-sampling type in nvjpeg library \n");
            return -1;
        }


        nvjpegImage_t imgdesc = {
            // .channel[4]
            {
                m_pBuffer_gpu,
                m_pBuffer_gpu + widths[0]*heights[0],
                m_pBuffer_gpu + widths[0]*heights[0]*2,
                m_pBuffer_gpu + widths[0]*heights[0]*3
            },
            // .pitch[4]
            {
                (unsigned int)(is_interleaved(oformat) ? widths[0] * 3 : widths[0]), // stride: x3 if interleaved
                (unsigned int)widths[0],
                (unsigned int)widths[0],
                (unsigned int)widths[0]
            }
        };

        imageSize = widths[0]*heights[0]*3;


        cudaDeviceSynchronize();
        chkErrors(cudaEventRecord(startEvent, NULL));

        nvjpegStatus_t eCode = nvjpegDecode(decoder_handle, jpeg_state,
                                            bitstream, bitstreamBytes,
                                            oformat,
                                            &imgdesc, NULL);
        if(eCode) {
            __E("nvjpegDecode error at %s:%d code=%d: \"%s\" \n",
                __FILE__, __LINE__,
                eCode,
                nvjpeg_status(eCode).c_str());
            return -1;
        }

        chkErrors(cudaEventRecord(stopEvent, NULL));
        chkErrors(cudaEventSynchronize(stopEvent));

        float msec_cost = 0.0f;
        chkErrors(cudaEventElapsedTime(&msec_cost, startEvent, stopEvent));
        __D("Decoder JPEG cost %f ms \n", msec_cost);

        if (img_dst) {
            chkErrors(
                cudaMemcpy(img_dst, m_pBuffer_gpu, imageSize, cudaMemcpyDeviceToHost));
        }

        return imageSize ? 0 : -1;
    };
};


class Nvjpeg_encoder
{
private:
    unsigned char* m_pBuffer_gpu;

public:
    cudaEvent_t startEvent, stopEvent;
    nvjpegEncoderParams_t encode_params;
    nvjpegHandle_t encoder_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegEncoderState_t encoder_state;
    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_RGB;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGB;
    nvjpegChromaSubsampling_t isampling; //nvjpeg YUV format
    nvjpegImage_t m_imgdesc;

    encode_params_t m_params;
    unsigned int m_widths[NVJPEG_MAX_COMPONENT]; // store width of every image channel
    unsigned int m_heights[NVJPEG_MAX_COMPONENT];
    size_t imageSize;
    int dev;

    std::vector<uint8_t> m_bitstream; // output jpeg bitstream
    size_t bitstreamBytes = 0;

    static int dev_malloc(void **p, size_t s) {
        return (int)cudaMalloc(p, s);
    }

    static int dev_free(void *p) {
        return (int)cudaFree(p);
    }

    bool is_interleaved(nvjpegOutputFormat_t format) {
        return (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI);
    }

    Nvjpeg_encoder(encode_params_t params) {
        m_params = params;
        std::cout<<m_params.input_w<<"  "<<m_params.input_h<< std::endl;
        init();
    };

    Nvjpeg_encoder() {
        encode_params_t params = {
            std::string("yuv"),
            std::string("420"),
            70, //compress quality
            0,
            0,
            1920,
            1080
        };

        m_params = params;
        init();
    };

    ~Nvjpeg_encoder() {
        // Free memory
        chkErrors(cudaFree(m_pBuffer_gpu));
        // free encode parameters
        chkErrors(nvjpegEncoderParamsDestroy(encode_params));
        chkErrors(nvjpegEncoderStateDestroy(encoder_state));
        chkErrors(nvjpegJpegStateDestroy(jpeg_state));
        chkErrors(nvjpegDestroy(encoder_handle));
        chkErrors(cudaEventDestroy(startEvent));
        chkErrors(cudaEventDestroy(stopEvent));
    };

    void init() {
        cudaDeviceProp props;
        chkErrors(cudaGetDeviceProperties(&props, m_params.dev));

        __D("Encoder Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
            m_params.dev, props.name, props.multiProcessorCount,
            props.maxThreadsPerMultiProcessor, props.major, props.minor,
            props.ECCEnabled ? "on" : "off");

        chkErrors(cudaSetDevice(m_params.dev));

        nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
        chkErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &encoder_handle));
        chkErrors(nvjpegJpegStateCreate(encoder_handle, &jpeg_state));
        chkErrors(nvjpegEncoderStateCreate(encoder_handle, &encoder_state, NULL));
        chkErrors(nvjpegEncoderParamsCreate(encoder_handle, &encode_params, NULL));

        // sample input parameters
        chkErrors(nvjpegEncoderParamsSetQuality(encode_params, m_params.quality, NULL));
        chkErrors(
            nvjpegEncoderParamsSetOptimizedHuffman(encode_params, m_params.huf, NULL));
        chkErrors(
            cudaEventCreate(&startEvent, cudaEventBlockingSync));
        chkErrors(
            cudaEventCreate(&stopEvent, cudaEventBlockingSync));

        processArgs();
    };

    int processArgs() {
        for(int i=0; i < NVJPEG_MAX_COMPONENT; i++)
        {
            m_widths[i] = m_params.input_w;
            m_heights[i] = m_params.input_h;
        }

        std::string fmt(m_params.format);
        std::string sSubsampling(m_params.subsampling);

        if (fmt == "yuv")
        {
            oformat = NVJPEG_OUTPUT_YUV;
            // isampling
        }
        else if (fmt == "rgb")
        {
            oformat = NVJPEG_OUTPUT_RGB;
            iformat = NVJPEG_INPUT_RGB;
        }
        else if (fmt == "bgr")
        {
            oformat = NVJPEG_OUTPUT_BGR;
            iformat = NVJPEG_INPUT_BGR;
        }
        else if (fmt == "rgbi")
        {
            oformat = NVJPEG_OUTPUT_RGBI;
            iformat = NVJPEG_INPUT_RGBI;
        }
        else if (fmt == "bgri")
        {
            oformat = NVJPEG_OUTPUT_BGRI;
            iformat = NVJPEG_INPUT_BGRI;
        }
        else
        {
            __E("Unknown or unsupported format: %s \n", &fmt[0]);
            return -1;
        }
        __I("format: %s \n", &fmt[0]);

        // TODO
        imageSize = m_widths[0] * m_heights[0] * 3;

        if (sSubsampling == "444")
        {
            isampling = NVJPEG_CSS_444;
        }
        else if (sSubsampling == "422")
        {
            isampling = NVJPEG_CSS_422;
        }
        else if (sSubsampling == "420")
        {
            isampling = NVJPEG_CSS_420;
        }
        else if (sSubsampling == "440")
        {
            isampling = NVJPEG_CSS_440;
        }
        else if (sSubsampling == "411")
        {
            isampling = NVJPEG_CSS_411;
        }
        else if (sSubsampling == "410")
        {
            isampling = NVJPEG_CSS_410;
        }
        else if (sSubsampling == "400")
        {
            isampling = NVJPEG_CSS_GRAY;
        }
        else
        {
            std::cerr << "Unknown or unsupported subsampling: " << sSubsampling << std::endl;
            return -1;
        }

        chkErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, isampling, NULL));
        chkErrors(cudaMalloc(&m_pBuffer_gpu, m_widths[0] * m_heights[0] * NVJPEG_MAX_COMPONENT));

        nvjpegImage_t imgdesc = {
            // .channel[4]
            {
                m_pBuffer_gpu,
                m_pBuffer_gpu + m_widths[0]*m_heights[0],
                m_pBuffer_gpu + m_widths[0]*m_heights[0]*2,
                m_pBuffer_gpu + m_widths[0]*m_heights[0]*3
            },
            // .pitch[4]
            {
                (unsigned int)(is_interleaved(oformat) ? m_widths[0] * 3 : m_widths[0]),
                (unsigned int)m_widths[0],
                (unsigned int)m_widths[0],
                (unsigned int)m_widths[0]
            }
        };

        m_imgdesc = imgdesc;

        __D("Init nvjpeg encoder parameters finished\n");
        return 0;
    };

    void feed_from_device(unsigned char *d_img_src) {
        chkErrors(cudaMemcpy(m_pBuffer_gpu, d_img_src, imageSize, cudaMemcpyDeviceToDevice));
    }

    void feed_from_host(unsigned char *h_img_src) {
        chkErrors(cudaMemcpy(m_pBuffer_gpu, h_img_src, imageSize, cudaMemcpyHostToDevice));
    }

    int encode_from_device(unsigned char *d_img_src) {
        feed_from_device(d_img_src);
        return encode();
    }

    int encode(unsigned char *img_src) {
        feed_from_host(img_src);
        return encode();
    }

    int encode() {
        cudaDeviceSynchronize();
        chkErrors(cudaEventRecord(startEvent, NULL));
        if (NVJPEG_OUTPUT_YUV == oformat)
        {
            chkErrors(nvjpegEncodeYUV(encoder_handle,
                                      encoder_state,
                                      encode_params,
                                      &m_imgdesc,
                                      isampling,
                                      m_widths[0],
                                      m_heights[0],
                                      NULL));
        }
        else
        {
            chkErrors(nvjpegEncodeImage(encoder_handle,
                                        encoder_state,
                                        encode_params,
                                        &m_imgdesc,
                                        iformat,
                                        m_widths[0],
                                        m_heights[0],
                                        NULL));
        }

        // later get_bits();

        chkErrors(cudaEventRecord(stopEvent, NULL));
        chkErrors(cudaEventSynchronize(stopEvent));
        float loopTime = 0;
        chkErrors(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
        double encoder_time = static_cast<double>(loopTime);
        //std::cout << "Encoder JPEG cost time: " << encoder_time << " ms" << std::endl;

        // std::string output_filename = "out.jpg";
        // std::cout << "Writing JPEG file: " << output_filename << std::endl;
        // std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
        // outputFile.write(reinterpret_cast<const char *>(m_bitstream.data()), static_cast<int>(bitstreamBytes));
        return 0; //bitstreamBytes ? 0 : -1;
    };

    int get_bits(uint32_t *numBytes = NULL, unsigned char *buffer = NULL) {
        chkErrors(nvjpegEncodeRetrieveBitstream(
                      encoder_handle,
                      encoder_state,
                      NULL,
                      &bitstreamBytes,
                      NULL));

        if (buffer == NULL) {
            m_bitstream.resize(bitstreamBytes);
            buffer = m_bitstream.data();
        }

        chkErrors(nvjpegEncodeRetrieveBitstream(
                      encoder_handle,
                      encoder_state,
                      buffer,
                      &bitstreamBytes,
                      NULL));

        if(numBytes) {
            *numBytes = bitstreamBytes;
        }

        return 0;
    }

    int encode_to_jpg(unsigned char *img_src, std::string jpg_path = "out.jpg") {
        encode(img_src);
        get_bits();

        auto outputFile = fopen(&jpg_path[0], "w+");
        if (fwrite(m_bitstream.data(), bitstreamBytes, 1, outputFile) != 1) {
            return -1;
        }
        return fclose(outputFile);
    }
};


#endif
