#include <trt_model.h>

// __global__ void increment_kernel(u_int8_t *parsing, u_int8_t *img, int resolution);
class RVM : public TRT_MODEL{
	public:
		int32_t max_size;
		RVM(const std::string &engine_filename);
		bool infer();
};

class HP : public TRT_MODEL{
	public:
		HP(const std::string &engine_filename);
		bool infer();
		int ThreadNumPerBlock, BlockNum;
		const int resolution = 3840 * 2160;
};