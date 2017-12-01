import os
import scipy.io as sio
import cv2
import numpy as np
DEBUG = False
SIZE = 224
PATH = '/home/zxy/Datasets/pascal3d/'
ANNOT_PATH = PATH + 'Annotations/'
IMAGE_PATH = PATH + 'Images/'
SAVE_PATH = PATH + 'Pascal3D/'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    
objs = os.listdir(ANNOT_PATH)

for obj in objs:
    obj_path = SAVE_PATH + obj
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)
    view_file = open(obj_path + '/annots.txt', 'w')
    img_path = IMAGE_PATH + obj
    #if not ('imagenet' in obj):
    #  continue
    for im_name in os.listdir(img_path):
        obj_id = im_name.split('.')[0]
        annot_file = ANNOT_PATH + obj + '/' + obj_id
        data = sio.loadmat(annot_file)
        data_objects = data['record']['objects'][0][0]
        cls = data_objects['class'][0]
        bbox = data_objects['bbox'][0]
        viewpoint = data_objects['viewpoint'][0]
        img = cv2.imread(img_path + '/' + im_name)
        
        #
        #print 'bbox', bbox, bbox.shape
        #print 'viewpoint', viewpoint, viewpoint.shape
        
        pad = max(img.shape[0], img.shape[1])
        pad_img = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2, 3), dtype = np.uint8)
        pad_img[pad:pad+img.shape[0], pad:pad+img.shape[1]] = img.copy()
        
        
        if DEBUG:
            cv2.imshow('img', img)
        cnt = 0
        for i in range(bbox.shape[0]):
            #print cls[i][0]
            if not (cls[i][0] in obj):
                continue
            try:
                box = bbox[i][0]
                l = int(max(box[2] - box[0], box[3] - box[1]) * 1.2)
                ct = [int(box[0] + box[2]) / 2 + pad, int(box[1] + box[3]) / 2 + pad]
                #print 'ct', ct
                #print 'l', l
                #print 'sz', pad_img.shape
                new_img = pad_img[ct[1] - l / 2:ct[1] + l / 2, ct[0] - l / 2:ct[0] + l / 2].copy()
                new_img = cv2.resize(new_img, (SIZE, SIZE))
                v = viewpoint[i][0]
                a = v['azimuth'][0][0][0]
                e = v['elevation'][0][0][0]
                save_name = '{}/{}_{}.png'.format(obj_path, obj_id, cnt)
                cv2.imwrite(save_name, new_img)
                view_file.write('{},{},{}\n'.format(obj_id, int(a), int(e)))
                cnt += 1
                #print 'saving', save_name
            except:
                print obj_path, obj_id, 'pass'
                continue

            #print 'a', viewpoint[i]['azimuth']
            #print 'a v', v['azimuth'][0]
            if DEBUG:
                cv2.imshow('new_img', new_img)
                cv2.waitKey()
                cv2.circle(img, (box[0], box[1]), 3, (255, 0, 0), -1)
                cv2.circle(img, (box[0], box[3]), 3, (255, 0, 0), -1)
                cv2.circle(img, (box[2], box[3]), 3, (255, 0, 0), -1)
                cv2.circle(img, (box[2], box[1]), 3, (255, 0, 0), -1)
                cv2.line(img, (box[0], box[1]), (box[0], box[3]), (255, 0, 0), 2)
                cv2.line(img, (box[0], box[1]), (box[2], box[1]), (255, 0, 0), 2)
                cv2.line(img, (box[2], box[3]), (box[0], box[3]), (255, 0, 0), 2)
                cv2.line(img, (box[2], box[3]), (box[2], box[1]), (255, 0, 0), 2)
        
        
        #break
    #break

    
    
