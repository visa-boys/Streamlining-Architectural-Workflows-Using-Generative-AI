from FloorplanModel import *
# merge all label into one multi-class label
floorplan_fuse_map = {
0: [  0,  0,  0], # background
1: [192,192,224], # closet
2: [192,255,255], # batchroom/washroom
3: [224,255,192], # livingroom/kitchen/dining room
4: [255,224,128], # bedroom
5: [255,160, 96], # hall
6: [255,224,224], # balcony
7: [224,224,224], # not used
8: [224,224,128], # not used
9: [255,60,128],  # extra label for opening (door&window)
10: [255,255,255]  # extra label for wall line
}
# use for index 2 rgb
floorplan_room_map = {
0: [  0,  0,  0], # background
1: [192,192,224], # closet
2: [192,255,255], # bathroom/washroom
3: [224,255,192], # livingroom/kitchen/diningroom
4: [255,224,128], # bedroom
5: [255,160, 96], # hall
6: [255,224,224], # balcony
7: [224,224,224], # not used
8: [224,224,128]  # not used
}
# boundary label
floorplan_boundary_map = {
0: [  0,  0,  0], # background
1: [255,60,128],  # opening (door&window)
2: [255,255,255]  # wall line
}
def init(config):
    if config.loadmethod == 'log':
        model = FloorplanModel()
        model.load_weights(config.weight)
    img = mpimg.imread(config.image)
    shp = img.shape
    img = tf.convert_to_tensor(img,dtype=tf.uint8)
    img = tf.image.resize(img,[512,512])
    img = tf.cast(img,dtype=tf.float32)
    img = tf.reshape(img,[-1,512,512,3])/255
    model.trainable = False
    model.vgg16.trainable = False
    return model,img,shp

def predict(model,img,shp):
    features = []
    feature = img
    for layer in model.vgg16.layers:
        feature = layer(feature)
        if layer.name.find('pool') != -1:
            features.append(feature)
    x = feature
    features = features[::-1]
    del model.vgg16
    gc.collect()
    
    featuresrbp = []
    for i in range(len(model.rbpups)):
        x = model.rbpups[i](x)+model.rbpcv1[i](features[i+1])
        x = model.rbpcv2[i](x)
        featuresrbp.append(x)
    logits_cw = tf.keras.backend.resize_images(model.rbpfinal(x),
                            2,2,'channels_last')
    
    x = features.pop(0)
    nLays = len(model.rtpups)
    for i in range(nLays):
        rs = model.rtpups.pop(0)
        r1 = model.rtpcv1.pop(0)
        r2 = model.rtpcv2.pop(0)
        f = features.pop(0)
        x = rs(x)+r1(f)
        x = r2(x)
        a = featuresrbp.pop(0)
        x = model.non_local_context(a,x,i)
        
    del featuresrbp
    logits_r = tf.keras.backend.resize_images(model.rtpfinal(x),
                2,2,'channels_last')
    del model.rtpfinal
    
    return logits_cw,logits_r
def fill_break_line(cw_mask):
    broken_line_h = np.array([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [1,0,0,0,1],
                            [0,0,0,0,0],
                            [0,0,0,0,0]], dtype=np.uint8)
    broken_line_h2 = np.array([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [1,1,0,1,1],
                            [0,0,0,0,0],
                            [0,0,0,0,0]], dtype=np.uint8)
    broken_line_v = np.transpose(broken_line_h)
    broken_line_v2 = np.transpose(broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v2)

    return cw_mask
def flood_fill(test_array, h_max=255):
    """
    fill in the hole 
    """
    test_array = test_array.squeeze()
    input_array = np.copy(test_array) 
    el = ndimage.generate_binary_structure(2,2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array),
                    structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = ndimage.generate_binary_structure(2,1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,
            ndimage.grey_erosion(output_array, size=(3,3),
            footprint=el))
    return output_array
def refine_room_region(cw_mask, rm_ind):
    label_rm, num_label = ndimage.label((1-cw_mask))
    new_rm_ind = np.zeros(rm_ind.shape)
    for j in range(1, num_label+1):  
        mask = (label_rm == j).astype(np.uint8)
        ys, xs, _ = np.where(mask!=0)
        area = (np.amax(xs)-np.amin(xs))*(np.amax(ys)-np.amin(ys))
        if area < 100:
            continue
        else:
            room_types, type_counts = np.unique(mask*rm_ind,
                    return_counts=True)
            if len(room_types) > 1:
                    room_types = room_types[1:] 
                    # ignore background type which is zero
                    type_counts = type_counts[1:] 
                    # ignore background count
            new_rm_ind += mask*room_types[np.argmax(type_counts)]

    return new_rm_ind
def ind2rgb(ind_im, color_map=floorplan_room_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im==i)] = rgb
    
    return rgb_im.astype(int)
def post_process(rm_ind,bd_ind,shp):
    hard_c = (bd_ind>0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind>0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)
    cw_mask = np.reshape(cw_mask,(*shp[:2],-1))
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask>=1] = 255

    # refine fuse mask by filling theW hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask//255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask,rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask.reshape(*shp[:2],-1)*new_rm_ind
    new_bd_ind = fill_break_line(bd_ind).squeeze()
    return new_rm_ind,new_bd_ind

def colorize(r,cw):
    cr = ind2rgb(r,color_map=floorplan_fuse_map)
    ccw= ind2rgb(cw,color_map=floorplan_boundary_map)
    return cr,ccw
def convert_one_hot_to_image(one_hot,dtype='float',act=None):
    if act=='softmax':
        one_hot = tf.keras.activations.softmax(one_hot)
    [n,h,w,c] = one_hot.shape.as_list()
    im=tf.reshape(tf.keras.backend.argmax(one_hot,axis=-1),
                  [n,h,w,1])
    if dtype=='int':
        im = tf.cast(im,dtype=tf.uint8)
    else:
        im = tf.cast(im,dtype=tf.float32)
    return im
def main_process(config):
    model,img,shp = init(config)
    if config.loadmethod == "log":
        logits_cw,logits_r = predict(model,img,shp)
    #     logits_r = tf.convert_to_tensor(logits_r)
    logits_r = tf.image.resize(logits_r,shp[:2])
    logits_cw = tf.image.resize(logits_cw,shp[:2])
    r = convert_one_hot_to_image(logits_r)[0].numpy()
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()

    if not config.colorize and not config.postprocess:
        cw[cw==1] = 9; cw[cw==2] = 10; r[cw!=0] = 0
        return (r+cw).squeeze()
    elif config.colorize and not config.postprocess:
        r_color,cw_color = colorize(r.squeeze(),cw.squeeze())
        return r_color+cw_color

    newr,newcw = post_process(r,cw,shp)
    if not config.colorize and config.postprocess:
        newcw[newcw==1] = 9; newcw[newcw==2] = 10; newr[newcw!=0] = 0
        return newr.squeeze()+newcw
    newr_color,newcw_color = colorize(newr.squeeze(),newcw.squeeze())
    result = newr_color+newcw_color
    print(shp,result.shape)

    return result
def convert_bounding_boxes(bboxes):
    converted_bboxes = []
    for bbox in bboxes:
        top_left, bottom_right = bbox
        top_left_x, top_left_y = top_left
        bottom_right_x, bottom_right_y = bottom_right
        converted_bbox = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        converted_bboxes.append(converted_bbox)
    return converted_bboxes
def result_orient(image_rgb):
    target_pixels = {
        (224, 255, 192): 'Living Room/Kitchen/Dining room',
        (255, 160, 96): 'Hall',
        (255, 224, 128): 'Bedroom',
        (192, 192, 224): 'Closet',
        (192, 255, 255): 'Bathroom/Washroom',
        (255, 224, 224): 'Balcony'
    }
    bounding_boxes = []
    labels = []
    for pixel_value, label in target_pixels.items():
        target_pixel = np.array(pixel_value)
        mask = cv2.inRange(image_rgb, target_pixel, target_pixel)
        _, labels_mask, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Get the bounding boxes for each region
        for i in range(1, stats.shape[0]):
            x, y, w, h, area = stats[i]
            if area > 0:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                bounding_boxes.append((top_left, bottom_right))
                labels.append(label)

    converted_bboxes=convert_bounding_boxes(bounding_boxes)
    return converted_bboxes,labels

# import argparse
# from misc import main
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# image_path="C://Users//kishore prashanth//Downloads//newyork//test//45780715.jpg"
# image = cv2.imread(image_path)
# postproc=True
# color=True
# args = argparse.Namespace(image=image_path,
#         weight='E://Blender//vectorize//log files//log//store//G',loadmethod='log',
#         postprocess=postproc,colorize=color,
#         save=None)
# result = main(args)
# if postproc==True and color==False:
#     result[result != 10] = 0
#     result[result == 10] = 1
#     np.save('array_file.npy', result)
# plt.imshow(result)
# plt.xticks([])
# plt.yticks([])
# plt.title("Result")
# plt.show()