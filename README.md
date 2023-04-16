# 基于多分辨率哈希编码神经辐射场的三维人脸重建

## tested on:
    WSL2:
        CUDA 11.8
        CUDNN 8.6.0
        TensorRT 8.6.0
    
    自行在WSL2上安装[COLMAP](https://github.com/colmap/colmap)

## 文件组织
    client (on Windows)
        --[instant-ngp](https://github.com/NVlabs/instant-ngp)
            --instant-ngp.exe
        --resource
        ...
    
## 运行
    Windows:
        start.bat
    
    WSL2:
        python run.py