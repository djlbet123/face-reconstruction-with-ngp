from flask import Flask, request
from flask.json import jsonify
import subprocess
import os
import re
import threading

cv = subprocess.check_output('''bash.exe -c "ifconfig eth0 | grep 'inet'"''',
            shell=True)

ip = re.search("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(cv,'utf-8')).group()

def release_cache():
    import os
    os.system("echo 3 > /proc/sys/vm/drop_caches")

def matting(data):
    name = os.path.join(data['path'], data['name'])
    new_name = name.split('.')[0] + "_train_matting.mp4" if data['train_flag'] else name.split('.')[0] + "_matting.mp4"
    subprocess.call(f"./trt/build/rvm ./trt/model.engine {name} {new_name} {data['train_flag']}", shell=True)
    release_cache()
    print("视频转换成功，输出到" + os.path.join(data['path'], new_name))

def parsing(data):
    #try:
    name = data['path'] + "/" + data['name']
    subprocess.call(f"python parsing/infer.py {name} {name[:-4]} {data['train_flag']}", shell=True)
    release_cache()
        
def calculate(data):
    name = data['name']
    output_path = os.path.join(data['path'], name[:-4])
    print("请等待colmap计算相机姿态，这通常需要几分钟时间")
    if os.path.exists(output_path):
        os.system(f'rm -rf {output_path}')    
    os.makedirs(output_path)
    subprocess.call(f"python ./colmap/scripts/python/colmap2nerf.py --video_in {os.path.join(data['path'], name)} --video_fps 3 --run_colmap --aabb_scale 16 --images {os.path.join(output_path, 'images')} \
                --out {os.path.join(output_path, 'transforms.json')} --colmap_db {os.path.join(output_path, 'colmap.db')} --text {os.path.join(output_path, 'colmap_text')}", shell=True)
    print("预处理完成，可进行神经辐射场训练")

waiting_list = []
mapping = {'matting' : matting, 'parsing' : parsing, 'calculate' : calculate}
lock = False
import time

def check():
    global lock
    while lock:
        time.sleep(10)
    lock = True
    data = waiting_list.pop(0)
    mapping[data['type']](data)
    print(data, " is done")
    print("still has ", len(waiting_list), " tasks")
    lock = False


if __name__ == "__main__":
    app = Flask(__name__)

    @app.route('/run', methods=['POST'])
    def run():
        data = request.form.to_dict()
        waiting_list.append(data)
        print(waiting_list)
        check()
        return 'success'

    app.run(ip, port=18924)
