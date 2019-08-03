'''
Author  : Zhengwei Li
Version : 1.0.0 
'''
import time
import cv2
import torch 
import pdb
import argparse
import numpy as np
import os 
from collections import OrderedDict
import torch.nn.functional as F
import sys

np.set_printoptions(threshold=sys.maxsize)

# sys.path.insert(0, './matting/pre_trained/erd_seg_matting/model/')

# parser = argparse.ArgumentParser(description='human matting')
# parser.add_argument('--model', default='./src/matting/pre_trained/erd_seg_matting/model/model_obj.pth', help='preTrained model')
# parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')

# args = parser.parse_args()

torch.set_grad_enabled(False)
INPUT_SIZE = 512

    
#################################
#----------------
# if args.without_gpu:
#     print("use CPU !")
#     device = torch.device('cpu')
# else:
device = None

flag = None

if torch.cuda.is_available():
    flag = 1
    n_gpu = torch.cuda.device_count()
    print("----------------------------------------------------------")
    print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
    print("----------------------------------------------------------")

    device = torch.device('cuda:0,1')
else:
    flag = 2
    device = torch.device('cpu')


#################################
#---------------
def load_model():
    print('Loading model from {}...'.format("./src/matting/pre_trained/erd_seg_matting/model/model_obj.pth"))
    if flag == 2:
        myModel = torch.load("./src/matting/pre_trained/erd_seg_matting/model/model_obj.pth", map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load("./src/matting/pre_trained/erd_seg_matting/model/model_obj.pth")

    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(args, image, net):

    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (INPUT_SIZE,INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0


    tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)
    # -----------------------------------------------------------------

    t0 = time.time()

    seg, alpha = net(inputs)

    # print((time.time() - t0))  

    if args.without_gpu:
        alpha_np = seg[0,0,:,:].cpu().data.numpy()
    else:
        alpha_np = seg[0,0,:,:].data.cpu().numpy()


    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)


    # -----------------------------------------------------------------
    
    # fg = np.multiply(fg_alpha[..., np.newaxis], image)


    # gray
    bg = image
    # bg_alpha = 1 - fg_alpha[..., np.newaxis]
    bg_alpha = fg_alpha


    split =  0
    bg_alpha[bg_alpha <= split] = 0
    bg_alpha[bg_alpha > split] = 255

    bg_alpha[bg_alpha == 255] = 1

    bg_alpha = 1 - bg_alpha

    

    bg_alpha = np.array(bg_alpha).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bg_alpha, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # print(sizes)
    if len(sizes) != 0:

        bg_alpha = np.zeros((output.shape))
        max_e = sizes.tolist().index(max(sizes.tolist()))
        bg_alpha[output == max_e + 1] = 255

    bg_alpha = 1 - bg_alpha
    bg_alpha[bg_alpha == 1] = 255

    # unique, counts = np.unique(bg_alpha, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    # print(bg_alpha)
    # exit()

    # bg_alpha = fg_alpha[..., np.newaxis]
    
    # split =  0

    # bg_alpha[bg_alpha <= split] = 0

    # print(bg_alpha)

    # bg_alpha[bg_alpha > split] = 255

    

    # bg_gray = np.multiply(bg_alpha, image)
    # bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    # bg[:,:,0] = bg_gray
    # bg[:,:,1] = bg_gray
    # bg[:,:,2] = bg_gray
    
    # -----------------------------------------------------------------
    # fg : color, bg : gray
    # bg_alpha = np.squeeze(bg_alpha)
    

    bg[:,:,0] = np.where(bg_alpha>0,bg_alpha,bg[:,:,0])
    bg[:,:,1] = np.where(bg_alpha>0,bg_alpha,bg[:,:,1])
    bg[:,:,2] = np.where(bg_alpha>0,bg_alpha,bg[:,:,2])

    # out = fg + bg

    # fg : color

    out = bg
    out[out<0] = 0
    out[out>255] = 255

    out = out.astype(np.uint8)

    return out


def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0)

    while(1):
        # get a frame
        ret, frame = videoCapture.read()
        frame = cv2.flip(frame,1)
        frame_seg = seg_process(args, frame, net)


        # show a frame
        cv2.imshow("capture", frame_seg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()

def main(args):

    myModel = load_model(args)
    # camera_seg(args, myModel)


# main(args)

# if __name__ == "__main__":
#     main(args)


