#include <TRT_MODEL.h>

class Logger : public nvinfer1::ILogger {
	public:
		explicit Logger(Severity severity = Severity::kWARNING)
		: reportable_severity_(severity) {
		}

		void log(Severity severity, char const* msg) noexcept override {
			if (severity > reportable_severity_) return;
			switch (severity) {
				case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR"; break;
				case Severity::kERROR:          std::cerr << "ERROR"; break;
				case Severity::kWARNING:        std::cerr << "WARNING"; break;
				case Severity::kINFO:           std::cerr << "INFO"; break;
				case Severity::kVERBOSE:        std::cerr << "VERBOSE"; break;
				default:                        std::cerr << "UNKNOWN"; break;
			}
			std::cerr << ": " << msg << std::endl;
		}

	private:
		Severity reportable_severity_;
};

TRT_MODEL::TRT_MODEL(const std::string &engine_filename){
    std::ifstream engine_file(engine_filename, std::ios::binary);
	if (engine_file.fail()) {
		std::cerr << "ERROR: engine file load failed" << std::endl;
	}

	engine_file.seekg(0, std::ifstream::end);
	auto fsize = engine_file.tellg();
	engine_file.seekg(0, std::ifstream::beg);

	std::vector<char> engine_data(fsize);
	engine_file.read(engine_data.data(), fsize);

	static Logger logger{Logger::Severity::kINFO};
	auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
		nvinfer1::createInferRuntime(logger));
	engine_.reset(
		runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr));
	if (engine_ == nullptr) {
		std::cerr << "ERROR: engine data deserialize failed" << std::endl;
	}

  	assert(engine_ != nullptr);
	context = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
}

int32_t TRT_MODEL::PrintEngineInfo() {
	assert(engine_ != nullptr);
	int32_t max_size = 0, sum;
	std::cout << "Engine" << std::endl;
	std::cout << " Name=" << engine_->getName() << std::endl;
	std::cout << " DeviceMemorySize=" << engine_->getDeviceMemorySize() / (1<< 20)
		<< " MiB" << std::endl;
	std::cout << " MaxBatchSize=" << engine_->getMaxBatchSize() << std::endl;

	std::cout << "Bindings" << std::endl;
	nb = engine_->getNbBindings();

	static auto datatype_names = std::map<nvinfer1::DataType, std::string>{
		{nvinfer1::DataType::kFLOAT, "FLOAT"},
		{nvinfer1::DataType::kHALF, "HALF"},
		{nvinfer1::DataType::kINT8, "INT8"},
		{nvinfer1::DataType::kINT32, "INT32"},
		{nvinfer1::DataType::kBOOL, "BOOL"},
		{nvinfer1::DataType::kUINT8, "UINT8"},
	};

	for (int32_t i = 0; i < nb; i++) {
		sum = 1;
		auto is_input = engine_->bindingIsInput(i);
		auto name = engine_->getBindingName(i);
		auto dims = engine_->getBindingDimensions(i);
		auto datatype = engine_->getBindingDataType(i);

		std::cout << " " << (is_input ? "Input[" : "Output[") << i << "]"
				<< " name=" << name << " dims=[";
		for (int32_t j = 0; j < dims.nbDims; j++) {
			if (i>=1 && i <=4)
				sum *= dims.d[j];
			std::cout << dims.d[j];
			if (j < dims.nbDims-1) std::cout << ",";
		}
		std::cout << "] datatype=" << datatype_names[datatype] << std::endl;
		max_size = std::max(max_size, sum);
	}
	return max_size;
}

bool TRT_MODEL::Allocate_binding(){
	assert(engine_ != nullptr);
	auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
		engine_->createExecutionContext());
	if (!context) {
		return false;
	}
	
	auto GetMemorySize = [](const nvinfer1::Dims &dims,
							const int32_t elem_size) -> int32_t {
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
			std::multiplies<int64_t>()) * elem_size;
	};

	auto nb = engine_->getNbBindings();

	// Allocate CUDA memory for all bindings
	for (int32_t i = 0; i < nb; i++) {
		bindings.push_back(nullptr);
		bindings_size.push_back(0);
		auto dims = engine_->getBindingDimensions(i);
		int32_t size, tmp;
		switch (engine_->getBindingDataType(i)){
			case nvinfer1::DataType::kUINT8: tmp = sizeof(u_int8_t); break;
			case nvinfer1::DataType::kFLOAT: tmp = sizeof(float); break;
			case nvinfer1::DataType::kINT32: tmp = sizeof(int32_t); break;
			default: break;
		}
		size = GetMemorySize(dims, tmp);
		if (cudaMalloc(&bindings[i], size) != cudaSuccess) {
			std::cerr << "ERROR: cuda memory allocation failed, size = " << size
			<< " bytes" << std::endl;
		return false;
		}
		bindings_size[i] = size;
	}

	if (cudaStreamCreate(&stream) != cudaSuccess) {
		std::cerr << "ERROR: cuda stream creation failed" << std::endl;
		return false;
	}
}

TRT_MODEL::~TRT_MODEL(){
	// Free CUDA resources
	auto nb = engine_->getNbBindings();
	for (int32_t i = 0; i < nb; i++) {
		cudaFree(bindings[i]);
	}
	cudaStreamDestroy(stream);
	engine_->destroy();
	bindings.clear();
	bindings.shrink_to_fit();
	bindings_size.clear();
	bindings_size.shrink_to_fit();
	std::cout << "model and bingdings released" << std::endl;
}