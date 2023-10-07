header ONLY API: Nvjpeg_codec.hpp

# sample
## decode and encode
```cpp
    auto image_file = "input.jpg";
    FILE *streamFile = fopen(&image_file[0], "rb");
    fseek(streamFile, 0, SEEK_END);
    auto bitstreamBytes = ftell(streamFile);
    fseek(streamFile, 0, SEEK_SET);

    std::vector<unsigned char> vBuffer(bitstreamBytes);
    auto bitstream = vBuffer.data();
    if (fread(bitstream, bitstreamBytes, 1, streamFile) != 1) {
        __E("Error JPEG file %s for %ld bytes \n", &image_file[0], bitstreamBytes);
        return -1;
    }


    auto decoder = TestJpgDecoderCreate(0);
    TestBitstreams bs;
    bs.bitstream = bitstream;
    bs.bitstreamBytes = bitstreamBytes;
    TestImage im;
    im.colorStd = TEST_IMAGE_YUV;
    im.img      = (void*)buffer;

    auto ret = TestJpgDecode(decoder, &bs, &im);
    if (ret) {
        return ret;
    }

    char i420_path[100];
    sprintf(i420_path, "i420_%dx%d.yuv", im.width, im.height);
    auto i420File = fopen(i420_path, "w+");

    if (fwrite(im.img, im.width*im.height*2, 1, i420File) != 1) {
        return -1;
    }
    fclose(i420File);

    auto encoder = TestJpgEncoderCreate(TEST_IMAGE_YUV, //TEST_IMAGE_RGBI,
                                         im.width, im.height,
                                         0,0);

    for(int i = 0; i < 10; i++)
    {
        TestBitstreams output_bs = {0};
        TestJpgEncode(encoder, &im, &output_bs);

        auto jpg_path = std::to_string(i) + "out.jpg";
        auto outputFile = fopen(&jpg_path[0], "w+");
        if (fwrite(output_bs.bitstream, output_bs.bitstreamBytes, 1, outputFile) != 1) {
            return -1;
        }
        fclose(outputFile);
    }

    TestJpgEncoderDestroy(encoder);
    TestJpgDecoderDestroy(decoder);

```
## testcase
```cpp
extern "C" void* TestJpgEncoderCreate(uint32_t inputFormat,
                                       uint32_t width,
                                       uint32_t height,
                                       uint8_t quality,
                                       uint8_t HW_id) {
    encode_params_t params = {
        std::string("rgbi"),
        std::string("422"),
        70, // compress quality
        0,  // buffman
        0, // gpu device id
        1920, // width
        1080  //height
    };

    if (inputFormat == TEST_IMAGE_YUV) {
        params.format = "yuv";
        params.subsampling = "420";
    } else if (inputFormat == TEST_IMAGE_RGBI) {
        params.format = "rgbi";
        params.subsampling = "422";
    } else if (inputFormat == TEST_IMAGE_RGB) {
        params.format = "rgb";
        params.subsampling = "422";
    }

    if (quality) {
        params.quality = (int)quality;
    }

    if (HW_id) {
        params.dev = (int) HW_id;
    }

    if (width && height) {
        params.input_w = width;
        params.input_h = height;
    }

    auto encoder = new Nvjpeg_encoder(params);
    return static_cast<void*>(encoder);
}

extern "C" int
TestJpgEncodeFeed(void *encoder_handle,
                   const TestImage *input_image,
                   uint8_t encoder_quality)
{
    auto encoder = static_cast<Nvjpeg_encoder*>(encoder_handle);
    int ret = 0;

    if (encoder_quality) {
        auto quality = (int)encoder_quality;
        chkErrors(
            nvjpegEncoderParamsSetQuality(encoder->encode_params,
                                          quality,
                                          NULL));
        encoder->m_params.quality = quality;
    }

    if (input_image) {
        auto h_img_src = (unsigned char*)input_image->img;
        auto d_img_src = (unsigned char*)input_image->imgPriv;

        if (h_img_src) {
            encoder->feed_from_host(h_img_src);
        } else if (d_img_src) {
            encoder->feed_from_device(d_img_src);
        }
    } else {
        __D("feed nothing \n");
        // encode last image of encoder
        // ret = encoder->encode();
    }

    return ret;
}


extern "C" int
TestJpgEncodeBits(void *encoder_handle)
{
    auto encoder = static_cast<Nvjpeg_encoder*>(encoder_handle);
    return encoder->encode();
}


extern "C" int
TestJpgEncodeGetBits(void *encoder_handle,
                      TestBitstreams *bitstreams)
{
    int ret = -1;
    auto encoder = static_cast<Nvjpeg_encoder*>(encoder_handle);
    if (bitstreams) {
        encoder->get_bits(&bitstreams->bitstreamBytes, bitstreams->bitstream);
        if (bitstreams->bitstream == NULL) {
            bitstreams->bitstream = encoder->m_bitstream.data();
        }

        ret = (bitstreams->bitstreamBytes) ? 0 : -1;
    } else {
        ret = encoder->get_bits();
    }

    return ret;
}


extern "C" int
TestJpgEncode(void *encoder_handle,
               const TestImage *input_image,
               TestBitstreams *bitstreams)
{
    int ret = TestJpgEncodeFeed(encoder_handle, input_image, 0);
    if (ret) {
        return ret;
    }

    ret = TestJpgEncodeBits(encoder_handle);
    if (ret) {
        return ret;
    }

    ret = TestJpgEncodeGetBits(encoder_handle, bitstreams);
    return ret;
}

extern "C" int
TestJpgEncode_buf(void *encoder_handle,
                   unsigned char *buf,
                   unsigned char *bitstream, uint32_t *bitstreamBytes)
{
    auto encoder = static_cast<Nvjpeg_encoder*>(encoder_handle);
    encoder->feed_from_host(buf);

    int ret = TestJpgEncodeBits(encoder_handle);
    if (ret) {
        return ret;
    }

    if (bitstream) {
        ret = encoder->get_bits(bitstreamBytes, bitstream);
    }

    return ret;
}

extern "C" void
TestJpgEncoderDestroy(void *encoder_handle) {
    auto encoder = static_cast<Nvjpeg_encoder*>(encoder_handle);
    delete(encoder);
}


extern "C" void* TestJpgDecoderCreate(uint8_t HW_id) {
    auto decoder = new Nvjpeg_decoder(HW_id);

    return static_cast<void*>(decoder);
}

extern "C" int TestJpgDecode_buf(void *decoder_handle,
                              unsigned char *bitstream, size_t bitstreamBytes,
                              unsigned char *img_dst)
{
    auto decoder = static_cast<Nvjpeg_decoder*>(decoder_handle);
    return decoder->decode(bitstream, bitstreamBytes, img_dst);
}

extern "C" int TestJpgDecode(void *decoder_handle,
                              const TestBitstreams *bitstreams,
                              TestImage *output_image)
{
    auto decoder = static_cast<Nvjpeg_decoder*>(decoder_handle);


    if (output_image->colorStd) {
        auto oformat = static_cast<nvjpegOutputFormat_t>(output_image->colorStd);
        if (decoder->oformat != oformat) {
            __D("convert to format: %d:%s \n", oformat,
                (oformat == NVJPEG_OUTPUT_YUV)  ? "yuv" :
                (oformat == NVJPEG_OUTPUT_Y)    ? "y" :
                (oformat == NVJPEG_OUTPUT_RGB)  ? "rgb" :
                (oformat == NVJPEG_OUTPUT_BGR)  ? "bgr" :
                (oformat == NVJPEG_OUTPUT_RGBI) ? "rgbi" :
                (oformat == NVJPEG_OUTPUT_BGRI) ? "bgri" :
                "unknown"
                );
            decoder->oformat = oformat;
        }
    }

    auto ret = decoder->decode(bitstreams->bitstream, bitstreams->bitstreamBytes,
                               (unsigned char*)output_image->img);
    if (ret) {
        return ret;
    }

    output_image->nChannels = decoder->nComponent;
    output_image->imgPriv   = (void*)decoder->m_pBuffer_gpu;
    output_image->width     = decoder->widths[0];
    output_image->height    = decoder->heights[0];
    output_image->colorStd  = static_cast<TestImageFormat_t>(decoder->oformat);

    return ret;
}

extern "C" int TestImageGetBits(const TestImage *image, uint8_t *img_dst) {
    auto w   = image->width;
    auto h   = image->height;
    auto fmt = image->colorStd;
    size_t imageSize = w*h*3;


    if (fmt == TEST_IMAGE_YUV) {
        imageSize = w * h * 1.5;
    } else if (fmt == TEST_IMAGE_Y) {
        imageSize = w * h * 1;
    } else if (fmt == TEST_IMAGE_RGBI) {
        imageSize = w * h * 3;
    }

    chkErrors(
        cudaMemcpy(img_dst, image->imgPriv, imageSize, cudaMemcpyDeviceToHost));

    return 0;
}

extern "C" void TestJpgDecoderDestroy(void *decoder_handle) {
    auto decoder = static_cast<Nvjpeg_decoder*>(decoder_handle);
    delete(decoder);
}

```
