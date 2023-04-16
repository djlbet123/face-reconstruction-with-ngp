#include <rvm_hp.h>

// __global__ void mask_kernel(u_int8_t *parsing, u_int8_t *img, int resolution){
// 	//parsing id 1, 2, 4, 13
// 	int idx = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (idx < resolution){
// 		bool flag = (parsing[idx] == 1) && (parsing[idx] == 2) && (parsing[idx] == 4) && (parsing[idx] == 13);
// 		if(flag){
// 			img[idx*3] = 255;
// 			img[idx*3+1] = 255;
// 			img[idx*3+2] = 255;
// 		}
// 	}
// }

RVM::RVM(const std::string &engine_filename):TRT_MODEL(engine_filename) {
	max_size = PrintEngineInfo();
}

bool RVM::infer() {
	// Run TensorRT inference
	bool status = context->enqueueV2(bindings.data(), stream, nullptr);
	if (!status) {
		std::cout << "ERROR: TensorRT inference failed" << std::endl;
		return false;
	}

	// update rec
	for (int j = 1; j <= 4; j++){
		std::swap(bindings[j], bindings[9-j]);
	}

	// get face
	//mask_kernel<<<BlockNum, ThreadNumPerBlock>>>((u_int8_t *) bindings[10], (u_int8_t*) bindings[9], resolution);

	return true;
}

HP::HP(const std::string &engine_filename):TRT_MODEL(engine_filename) {
	PrintEngineInfo();
}

bool HP::infer() {
	// Run TensorRT inference
	bool status = context->enqueueV2(bindings.data(), stream, nullptr);
	if (!status) {
		std::cout << "ERROR: TensorRT inference failed" << std::endl;
		return false;
	}
	// get face
	//mask_kernel<<<BlockNum, ThreadNumPerBlock>>>((u_int8_t *) bindings[10], (u_int8_t*) bindings[9], resolution);
	return true;
}