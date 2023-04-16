#ifndef _TRT_MODEL_
#define _TRT_MODEL_

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

class TRT_MODEL{
	public:
		std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        std::vector<void *> bindings;
        std::vector<int32_t> bindings_size;
        cudaStream_t stream;


        TRT_MODEL(const std::string &engine_filename); // load model
        ~TRT_MODEL();
		int32_t PrintEngineInfo();
        int32_t nb;
        bool Allocate_binding();

};

#endif