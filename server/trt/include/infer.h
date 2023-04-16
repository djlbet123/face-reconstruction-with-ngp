#include <rvm_hp.h>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define N_ARG 4

class INFER{
    public:
        INFER(std::vector<std::string> &arg_s);
        ~INFER();
        void run();
        void print_devide_prop();
    
    private:
        cv::VideoCapture *cap;
        cv::VideoWriter *out;
        cv::Mat src, com, com_out, src_in;
        std::unique_ptr<RVM> model;
        int frame_num;
        int step;
        bool train_flag;
        std::string path;
        void load_img_copy_to_device();
        void copy_to_host_write_img();
};