import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import torch
import cv2
import sys
import numpy as np
import time

def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image

    int_lmrks = np.array(image_landmarks, dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    if image_landmarks.shape[0] == 68:
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[0:9],
                            int_lmrks[17:18])))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[8:17],
                            int_lmrks[26:27])))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[17:20],
                            int_lmrks[8:9])))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[24:27],
                            int_lmrks[8:9])))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[19:25],
                            int_lmrks[8:9],
                            )))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[17:22],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            )))), (1,))
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(
            np.concatenate((int_lmrks[22:27],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            )))), (1,))
        cv2.fillConvexPoly(
            hull_mask, np.int32(cv2.convexHull(int_lmrks[27:36])), (1,))
    
    else:
        cv2.fillConvexPoly(hull_mask, np.int32(cv2.convexHull(int_lmrks
                            )), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    
    return hull_mask

def get_seg_face(img_path, new_name, train_flag):
    begin = time.time()
    cap = cv2.VideoCapture(img_path)
    fps = 4 if train_flag else 30
    out = cv2.VideoWriter(new_name+"_face.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    box = cv2.VideoWriter(new_name+"_box.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    landmark_video = cv2.VideoWriter(new_name+"_landmarks.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    torchlm.runtime.bind(faceboxesv2(device="cuda"))  # set device="cuda" if you want to run with CUDA
    # set map_location="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
    pipnet(backbone="resnet101", pretrained=True,  
            num_nb=10, num_lms=98, net_stride=32, input_size=256,
            meanface_type="wflw", map_location="cuda", checkpoint=None) 
    ) # will auto download pretrained weights from latest release if pretrained=True
    
    
    flag, image = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    step = 1
    # if train_flag:
    #     if (count / 120.0 >= 2):
    #         step = count / 120.0
    #     if (count / 80.0 >= 2):
    #         step = count / 80.0
    #     elif (count / 60.0 >= 2):
    #         step = count / 60.0
    #     else:
    #         step = 1;    

    # elif count < 2000:
    #     step = 1
    
    step = int(step)
    while flag:
        another = np.copy(image)
        landmarks, bboxes = torchlm.runtime.forward(image)
        #cv2.imwrite("out_" + name, image)
        for landmark in landmarks:
            mask = get_image_hull_mask(np.shape(image), landmark).astype(np.bool_).squeeze(-1)
            image[~mask] = (255, 255, 255)
            another = torchlm.utils.draw_bboxes(another, bboxes=bboxes)
            another_2 = torchlm.utils.draw_landmarks(another, landmarks=landmarks)
            box.write(another)
            landmark_video.write(another_2)
            out.write(image)
        for i in range(1,step):
            cap.grab()
        flag, image = cap.read()
        print(idx, "/", count)
        idx += step
    
    print(time.time() - begin)
    cap.release()
    out.release()
    box.release()
    landmark_video.release()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2], sys.argv[3])
    get_seg_face(sys.argv[1], sys.argv[2], sys.argv[3]=='True')
    
