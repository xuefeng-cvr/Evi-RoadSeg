import os
import numpy as np
import cv2
import json


def Disp2depth(fx, baseline, disp, disp_path=''):
    # fx = 2268.36
    # baseline = 0.222126
    delta = 256
    if(disp is None):
        print("error!!! \n")
        print(disp_path)
        print("\n")
    try:
        disp_mask = disp > 0
    except:
        print("error!!! \n")
        print(disp_path)
        print("\n")
    depth = disp.astype(np.float32)
    depth[disp_mask] = (depth[disp_mask] - 1) / delta
    disp_mask = depth > 0
    depth[disp_mask] = fx * baseline / depth[disp_mask]
    return depth


def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        calib_info = json.load(f)
        baseline = calib_info['extrinsic']['baseline']
        fx = calib_info['intrinsic']['fx']
        fy = calib_info['intrinsic']['fy']
        u0 = calib_info['intrinsic']['u0']
        v0 = calib_info['intrinsic']['v0']
    return baseline, fx, fy, u0, v0

# shadow augmentation
def add_ellipse_shadow(img, label):
    h, w, _ = img.shape

    # calculate position
    sum_r = np.sum(label, 1)
    index_r = np.where(sum_r > 0)[0]
    random_r = np.random.randint(index_r[10], index_r[-1])

    index_c = np.where(label[random_r, :].squeeze() > 0)[0]
    random_c = np.random.randint(index_c[0]-50, index_c[-1]+50)

    e_h = np.random.randint(20, 100)
    e_w = np.random.randint(50, 400)

    weight = 0.1 + np.random.rand()*0.6

    print('random_r', random_r)
    print('random_c', random_c)

    print('e_h', e_h)
    print('e_w', e_w)

    print('weight', weight)

    mask = 255*np.ones(img.shape).astype(np.uint8)
    mask = cv2.ellipse(mask, (random_c, random_r), (e_w, e_h), 0, 0, 360, (0, 0, 0), -1)

    background = mask[:,:,1] == 255
    img_aug = cv2.addWeighted(img, 1 - weight, mask, weight, 0)
    img_aug[background] = img[background]

    return img_aug


def add_tree_shadow(img, label, tree_shadow):
    h_, w_, _ = img.shape

    if np.sum(label) < 10000:
        return img

    # calculate position
    sum_r = np.sum(label, 1)
    index_r = np.where(sum_r > 0)[0]
    random_r = np.random.randint(index_r[10], index_r[-1])

    if np.sum(label[random_r, :]) < 10:
        return img

    index_c = np.where(label[random_r, :].squeeze() > 0)[0]
    random_c = np.random.choice([index_c[0], index_c[-1]])

    if random_c == index_c[-1]:
        tree_shadow = cv2.flip(tree_shadow, 1)

    # calculate size
    h, w = tree_shadow.shape
    scale = [1, 2, 4, 6]
    scale = np.random.choice(scale)
    dist = (random_r - index_r[0]) / (index_r[-1] - index_r[0])
    size = (int(w * dist * scale), int(h * dist * scale))
    if not (size[0] > 0 and size[1] > 0):
        return img
    tree_shadow = cv2.resize(tree_shadow, size, interpolation=cv2.INTER_NEAREST)
    h, w = tree_shadow.shape

    weight = 0.3 + np.random.rand() * 0.5

    # calculate img and shadow coordinate
    if random_r - h // 2 > index_r[0] - 20:
        i_h0 = random_r - h // 2
        s_h0 = 0
    else:
        i_h0 = index_r[0] - 20
        s_h0 = h // 2 - (random_r - (index_r[0] - 20))

    if random_r + h // 2 < h_:
        i_h1 = random_r + h // 2
        s_h1 = h - h % 2
    else:
        i_h1 = h_
        s_h1 = h // 2 + h_ - random_r

    if random_c - w // 2 > 0:
        i_w0 = random_c - w // 2
        s_w0 = 0
    else:
        i_w0 = 0
        s_w0 = w // 2 - random_c

    if random_c + w // 2 < w_:
        i_w1 = random_c + w // 2
        s_w1 = w - w % 2
    else:
        i_w1 = w_
        s_w1 = w // 2 + w_ - random_c
    
    if i_h1 - i_h0 <= 0 or i_w1 - i_w0 <= 0:
        return img

    img_crop = img[i_h0:i_h1, i_w0:i_w1, :]
    shadow_crop = tree_shadow[s_h0:s_h1, s_w0:s_w1]

    mask = 255*np.ones(img_crop.shape).astype(np.uint8)
    mask[:, :, 0] = 255 - shadow_crop
    mask[:, :, 1] = 255 - shadow_crop
    mask[:, :, 2] = 255 - shadow_crop

    background = mask[:,:,1] == 255
    result = cv2.addWeighted(img_crop, 1 - weight, mask, weight, 0)
    result[background] = img_crop[background]

    img_aug = img.copy()
    img_aug[i_h0:i_h1, i_w0:i_w1, :] = result

    return img_aug

# augmentation
def geometry_depth_aug(img, depth, label, cityscapes_cars, instance_bottom_depth, cityscapes_root):

    if np.sum(label) < 500:
        return img, depth, label
    kernel = np.ones((24, 24), np.uint8)
    label_erode = cv2.erode(label, kernel)
    label_erode[np.where(depth == 0)] = 0
    img_depth_mean = (depth * label_erode).sum(1) / (label_erode.sum(1) + 0.00001)
    instance_bottom_depth = np.array(instance_bottom_depth)
    instance_id1 = np.where(instance_bottom_depth>np.min(img_depth_mean))
    instance_id2 = np.where(instance_bottom_depth<np.max(img_depth_mean))
    instance_id = np.intersect1d(instance_id1, instance_id2)
    n = len(instance_id)
    if n == 0:
        return img, depth, label
    random_car = np.random.randint(0, n)
    car_image_path = cityscapes_cars[instance_id[random_car]]
    instance_filename = car_image_path.split('/')[-1]
    folder = instance_filename.split('_')[0]
    instid = instance_filename.split('_')[-1][:5]
    instance_img_name = instance_filename[:-21]
    car_disp_path = os.path.join(cityscapes_root, 'disparity_trainvaltest/disparity/train/', folder, instance_img_name+'disparity.png')
    car_calib_path = os.path.join(cityscapes_root, 'camera_trainvaltest/camera/train', folder, instance_img_name+'camera.json')
    car_semantic_path = os.path.join(cityscapes_root, 'gtFine_trainvaltest/gtFine/train', folder, instance_img_name+'gtFine_labelIds.png')
    car_label_path = os.path.join(cityscapes_root, 'gtFine_trainvaltest/gtFine/train', folder, instance_img_name+'gtFine_instanceIds.png')
    car_image_path = os.path.join(cityscapes_root, 'leftImg8bit_trainvaltest/leftImg8bit/train', folder, instance_img_name+'leftImg8bit.png')
    car_image = cv2.cvtColor(cv2.imread(car_image_path), cv2.COLOR_BGR2RGB)

    semantic_label = cv2.imread(car_semantic_path, cv2.IMREAD_GRAYSCALE)
    semantic_label[semantic_label != 7] = 0
    semantic_label[semantic_label == 7] = 1

    if np.sum(semantic_label) < 1000:
        return img, depth, label
        
    car_disp_image = cv2.imread(car_disp_path, cv2.IMREAD_ANYDEPTH)
    baseline, fx, fy, u0, v0 = read_calib_file(car_calib_path)
    ExObj_depth = Disp2depth(fx, baseline, car_disp_image)

    car_label = cv2.imread(car_label_path, cv2.IMREAD_ANYDEPTH)
    car_image = cv2.resize(car_image, (1280, 640))
    car_label = cv2.resize(car_label, (1280, 640))
    ExObj_depth = cv2.resize(ExObj_depth, (1280, 640))
    h_, w_, _ = img.shape


    instid = int(instid)
    a = np.where(car_label == instid)
    bbox = [
        np.min(a[0]),
        np.min(a[1]),
        np.max(a[0]) + 1,
        np.max(a[1]) + 1,
    ]

    car_image = car_image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    ExObj_depth = ExObj_depth[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    car_label = (car_label == instid)[bbox[0]:bbox[2], bbox[1]:bbox[3]].astype(np.uint8)
    
    depthdelta = np.abs(img_depth_mean - instance_bottom_depth[instance_id[random_car]])
    row = np.argmin(depthdelta)

    if np.sum(label_erode[row, :]) < 10:
        return img, depth, label

    index_c = np.where(label_erode[row, :].squeeze() > 0)[0]
    column = np.random.randint(index_c[0], index_c[-1])

    h, w = car_label.shape

    # calculate img and car coordinate
    if row - h > 0:
        i_h0 = row - h
        c_h0 = 0
    else:
        i_h0 = 0
        c_h0 = h - row

    i_h1 = row
    c_h1 = h

    if column - w // 2 > 0:
        i_w0 = column - w // 2
        c_w0 = 0
    else:
        i_w0 = 0
        c_w0 = w // 2 - column

    if column + w // 2 < w_:
        i_w1 = column + w // 2
        c_w1 = w - w % 2
    else:
        i_w1 = w_
        c_w1 = w // 2 + w_ - column

    img_crop = img[i_h0:i_h1, i_w0:i_w1, :].copy()
    depth_crop = depth[i_h0:i_h1, i_w0:i_w1].copy()
    label_crop = label[i_h0:i_h1, i_w0:i_w1].copy()
    car_image_crop = car_image[c_h0:c_h1, c_w0:c_w1, :].copy()
    ExObj_depth_crop = ExObj_depth[c_h0:c_h1, c_w0:c_w1].copy()
    car_label_crop = car_label[c_h0:c_h1, c_w0:c_w1].copy()


    img_crop[car_label_crop>0, :] = car_image_crop[car_label_crop>0, :]
    depth_crop[car_label_crop>0] = ExObj_depth_crop[car_label_crop>0]
    label_crop[car_label_crop>0] = 0

    img_aug = img.copy()
    img_aug[i_h0:i_h1, i_w0:i_w1, :] = img_crop

    depth_aug = depth.copy()
    depth_aug[i_h0:i_h1, i_w0:i_w1] = depth_crop
    depth_aug[depth == 0] = 0

    label_aug = label.copy()
    label_aug[i_h0:i_h1, i_w0:i_w1] = label_crop
    return img_aug, depth_aug, label_aug
