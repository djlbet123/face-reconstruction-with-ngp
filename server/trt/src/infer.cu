#include <infer.h>

void INFER::print_devide_prop(){
	int dev = 0;
	cudaDeviceProp devProp;
	if(cudaGetDeviceProperties(&devProp, dev) != cudaSuccess){
		std::cout<< "get devide failed" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大blocks数量: " << devProp.maxBlocksPerMultiProcessor << std::endl;
	// model->ThreadNumPerBlock = devProp.maxThreadsPerBlock;
	// model->BlockNum = devProp.maxBlocksPerMultiProcessor * devProp.multiProcessorCount;
}

INFER::INFER(std::vector<std::string> &arg_s){
	// open model and allocate bingings
	std::cout << "begin loading" << std::endl;
	model = std::make_unique<RVM>(arg_s[0]);
	std::cout << "load model succeed" << std::endl;

	int fps = 30;
	step = 4;

	//allocate binding
	model->Allocate_binding();
	std::cout << "allocate bindings succeed" << std::endl;

	// video
	cap = new cv::VideoCapture(arg_s[1]);
	out = new cv::VideoWriter(arg_s[2], cv::VideoWriter::fourcc('A','V','C','1'), fps, cv::Size(cap->get(3), cap->get(4)), true);
	frame_num = cap->get(cv::CAP_PROP_FRAME_COUNT);
	std::cout << "load video succeed" << std::endl;

	if (arg_s[3] == std::string("True")){
		fps = 4;
		if (frame_num / 120.0 >= 2)
			step = frame_num / 120.0;
		else if (frame_num / 80.0 >= 2)
			step = frame_num / 80.0;
		else if (frame_num / 60.0 >= 2)
			step = frame_num / 60.0;
		else
			step = 1;
		std::cout << "frame num is " << frame_num << ", step is " << step << std::endl;
	}

	// init r1i, r2i, r3i, r4i
	cv::Mat ret = cv::Mat::zeros(1, model->max_size, CV_32FC1);
	for (int32_t i = 1; i < 5; i++) {
		if (cudaMemcpyAsync(model->bindings[i], ret.data, model->bindings_size[i],
			cudaMemcpyHostToDevice, model->stream) != cudaSuccess) {
			std::cerr << "ERROR: CUDA memory copy of r" << i << "i failed, size = "
				<< model->bindings_size[i] << " bytes" << std::endl;
		}
	}

	com = cv::Mat(2160, 3840, CV_8UC3);
	src_in = cv::Mat(2160, 3840, CV_8UC3);
	com_out = cv::Mat(cap->get(4), cap->get(3), CV_8UC3);
	std::cout << "all loaded" << std::endl;
	print_devide_prop();
}

INFER::~INFER(){
	cap->release();
	out->release();
	std::cout << "INFER class released" << std::endl;
}

void INFER::run(){
	int size = (int)(frame_num / step);
	std::cout << frame_num << " " << step << " " << size <<std::endl; 
	for(int i = 0; i < size; ++i){
		load_img_copy_to_device();
		model->infer();
		copy_to_host_write_img();
		for(int j = 1; j<step; j++)
			cap->grab();
		std::cout << i << "/" << size << std::endl;
	}
}

void INFER::load_img_copy_to_device(){
	cap->read(src);
	cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
	auto in_ptr = &src;
    if (2160 != src.rows || 3840 != src.cols) {
      cv::resize(src, src_in, cv::Size(3840, 2160));
	  in_ptr = &src_in;
    }
	if (cudaMemcpyAsync(model->bindings[0], in_ptr->data, model->bindings_size[0],
		cudaMemcpyHostToDevice, model->stream) != cudaSuccess) {
		std::cerr << "ERROR: CUDA memory copy of src failed, size = "
			<< model->bindings_size[0] << " bytes" << std::endl;
		return;
	}
}

void INFER::copy_to_host_write_img(){
	// cv::Mat parsing_result = cv::Mat(2160, 3840, CV_8UC1);
	// Copy data from output binding memory
	if (cudaMemcpyAsync(com.data, model->bindings[model->nb-1], model->bindings_size[model->nb-1],
		cudaMemcpyDeviceToHost, model->stream) != cudaSuccess) {
		std::cerr << "ERROR: CUDA memory copy of output failed, size = "
			<< model->bindings_size[model->nb-1] << " bytes" << std::endl;
		return;
	}
    if (cap->get(4) != com.rows || cap->get(3) != com.cols) {
      cv::resize(com, com_out, cv::Size(cap->get(3), cap->get(4)));
	  cv::cvtColor(com_out, com_out, cv::COLOR_RGB2BGR);
    }
	else{
		cv::cvtColor(com, com_out, cv::COLOR_RGB2BGR);
	}
	out->write(com_out);
}

int main(int argc, char const *argv[]) {
	auto begin = std::chrono::high_resolution_clock::now(); //记录当前时间
	std::string arg_name[N_ARG] = {"engine_path", "input_video_path", "output_imgs_path", "train_flag"};
	std::vector<std::string> arg_s;
	for (int i = 0; i < N_ARG; i++){
		if (argc < i + 1){
			std::cout << "need another arg: " << arg_name[i] << std::endl;
			return EXIT_FAILURE;
		}
		arg_s.push_back(std::string(argv[i+1]));
		std::cout << arg_name[i] << " is " << arg_s[i] << std::endl; 
	}

	INFER infer(arg_s);
	infer.run();

	std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - begin;
	std::cout << duration.count() << "s" << std::endl; //输出运行时间
	return EXIT_SUCCESS;
}


// 使用GPU device 0: NVIDIA GeForce RTX 3060 Laptop GPU
// SM的数量：30
// 每个线程块的共享内存大小：48 KB
// 每个线程块的最大线程数：1024
// 每个SM的最大线程数：1536
// 每个SM的最大线程束数：48
// Bindings
//  Input[0] name=src dims=[1,2160,3840,3] datatype=UINT8
//  Input[1] name=r1i dims=[1,16,135,240] datatype=FLOAT
//  Input[2] name=r2i dims=[1,20,68,120] datatype=FLOAT
//  Input[3] name=r3i dims=[1,40,34,60] datatype=FLOAT
//  Input[4] name=r4i dims=[1,64,17,30] datatype=FLOAT
//  Output[5] name=r4o dims=[1,64,17,30] datatype=FLOAT
//  Output[6] name=r3o dims=[1,40,34,60] datatype=FLOAT
//  Output[7] name=r2o dims=[1,20,68,120] datatype=FLOAT
//  Output[8] name=r1o dims=[1,16,135,240] datatype=FLOAT
//  Output[9] name=com dims=[1,2160,3840,3] datatype=UINT8
//  Output[10] name=parsing_result dims=[1,2160,3840] datatype=UINT8