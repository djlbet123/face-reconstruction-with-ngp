# 基于多分辨率哈希编码神经辐射场的三维人脸重建

## Pipeline
![image](https://github.com/djlbet123/face-reconstruction-with-ngp/blob/master/img/pipeline.jpg)

## Result
![image](https://github.com/djlbet123/face-reconstruction-with-ngp/blob/master/img/result.jpg)

## Requirement(Tested):
    WSL2:
        CUDA 11.8
        CUDNN 8.6.0
        TensorRT 8.6.0
    
    Windows:
        instant-ngp
        colmap

    
## 文件组织
    client (on Windows)
        --instant-ngp(自行下载)
            --instant-ngp.exe
        --resource
        ...

## 运行
    Windows:
        start.bat
    
    WSL2:
        python run.py

## 使用的开源项目
[COLMAP](https://github.com/colmap/colmap)
[instant-ngp](https://github.com/NVlabs/instant-ngp)
[torchml](https://github.com/DefTruth/torchlm)

## 不足
该项目有部分目的是探索WSL作为云端服务器的可行性，所以参杂了pytorch,onnx,TensorRT各个框架，效率堪忧。经测试COLMAP在WSL并不支持GPU加速，所以仍在Windows上运行，instant-ngp在WSL不支持DLSS，也在Windows上运行。如果服务器是原生Ubuntu系统，全用TensorRT及C++实现，才能实现高效率云端计算。