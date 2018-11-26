from __future__ import division
import numpy as np
import os
import cv2

project_path = "E:/redraw/"
data_path = os.path.join(project_path, "data/")
image_path = os.path.join(project_path, "images/")
colordata_path = os.path.join(project_path, "colordata/")


image_width = 922
image_height = 691
CUT_NUMBER = 5
IMAGE_NUMBER = 30
IMAGE_ORDER = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
COLOR_ALL = [[0,0,0],[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
COLOR_DATA = []
VERTEXES_DATA = []
CIRCLE_SIZE = 3
CIRCLE_COLOR = [0,0,255]
PROPOTION = [2,3,4,5,6,7,8]    
VERTEXES_PERPOLYGEN = 4
CLICK_NUMBER = 0
CLICK_POINTS = []
CONFIRM_POINTS = []
CLICK_COLOR = []

def draw_polygen(vertexes, img, color):

    cv2.fillConvexPoly(img, vertexes, color = color)
    return img

def possible_cutpos(vertexes):
    possible_points = []
    
    for i in range(len(vertexes)):
        min_x = min(vertexes[i][0][0], vertexes[i][2][0])
        max_x = max(vertexes[i][0][0], vertexes[i][2][0])
        min_y = min(vertexes[i][0][1], vertexes[i][2][1])
        max_y = max(vertexes[i][0][1], vertexes[i][2][1])
        len_xside = max_x - min_x
        len_yside = max_y - min_y
        possible_cutpoints = []             
        for k in range(len(PROPOTION)):
            possible_cutpoints.append([min_x + PROPOTION[k]/10 * len_xside, min_y])
            possible_cutpoints.append([min_x, min_y + PROPOTION[k]/10 * len_yside])
            possible_cutpoints.append([min_x + PROPOTION[k]/10 * len_xside, max_y])
            possible_cutpoints.append([max_x, min_y + PROPOTION[k]/10 * len_yside])
        possible_points.append(possible_cutpoints)
    return possible_points            

def select_cutshape(vertexes, cut_points):
    midpoint = [0.5*(cut_points[0][0] + cut_points[1][0]), 0.5*(cut_points[0][1] + cut_points[1][1])]
    for i in range(len(vertexes)):   
        min_x = min(vertexes[i][0][0], vertexes[i][2][0])
        max_x = max(vertexes[i][0][0], vertexes[i][2][0])
        min_y = min(vertexes[i][0][1], vertexes[i][2][1])
        max_y = max(vertexes[i][0][1], vertexes[i][2][1])
        if (midpoint[1] > min_y) and (midpoint[1] < max_y) and (midpoint[0] > min_x) and (midpoint[0] < max_x):
            selected_shape = i 
            break
    return selected_shape
    
def vertexes_clustery(vertexes, vertexes_number, selected_shape, cut_points, vertexes_newshape1, vertexes_newshape2):
    selected_shape = select_cutshape(vertexes, cut_points)
    if cut_points[0][0] == cut_points[1][0]:
        if vertexes[selected_shape][vertexes_number-1][0] > cut_points[0][0]:
            vertexes_newshape1.append(vertexes[selected_shape][vertexes_number-1]) 
        if vertexes[selected_shape][vertexes_number-1][0] < cut_points[0][0]:
            vertexes_newshape2.append(vertexes[selected_shape][vertexes_number-1])
    else:    
        if vertexes[selected_shape][vertexes_number-1][1] > cut_points[0][1]:
            vertexes_newshape1.append(vertexes[selected_shape][vertexes_number-1]) 
        if vertexes[selected_shape][vertexes_number-1][1] < cut_points[0][1]:
            vertexes_newshape2.append(vertexes[selected_shape][vertexes_number-1])
            
    if vertexes_number == 1:
        vertexes_newshape1.extend(cut_points)
        vertexes_newshape2.extend(cut_points)
        return vertexes_newshape1, vertexes_newshape2
        
    return vertexes_clustery(vertexes, vertexes_number-1, selected_shape, cut_points,  vertexes_newshape1, vertexes_newshape2)

def points_sort(points):
    points_sorted = []
    points_number = len(points)
    center = (np.array(points)).sum(axis = 0)/points_number
    theta_original = []
    
    for x in xrange(points_number):
        if points[x][0] < center[0]:
            theta = np.arctan((points[x][1]-center[1])/(points[x][0]-center[0]))+np.pi            
        elif points[x][0] == center[0]:
            theta = 0.5*np.pi*((points[x][1]-center[1])/abs(points[x][1]-center[1]))            
        else:
            theta = np.arctan((points[x][1]-center[1])/(points[x][0]-center[0])) 
        theta_original.append(theta)

    thetaindex_sorted = np.argsort(theta_original)
    for x in xrange(len(points)):
        points_sorted.append(points[thetaindex_sorted[x]])
    return points_sorted     
    
def points_reallocation(cut_points, vertexes, selected_shape):
    vertexes_number = len(vertexes[selected_shape])
    vertexes_newshape1 = []
    vertexes_newshape2 = []
    vertexes1, vertexes2 = vertexes_clustery(vertexes = vertexes, vertexes_number = vertexes_number, cut_points = cut_points, selected_shape = selected_shape, vertexes_newshape1 = vertexes_newshape1, vertexes_newshape2 = vertexes_newshape2)   
    vertexes_leftbottom = points_sort(vertexes1)
    vertexes_righttop = points_sort(vertexes2)
    vertexes[selected_shape] = vertexes_leftbottom
    vertexes.append(vertexes_righttop)    
       
def mouse_select(event, x, y, flag, param):
    global CLICK_NUMBER, CLICK_POINTS, CONFIRM_POINTS 
    if event == cv2.EVENT_LBUTTONDOWN:
        if CLICK_NUMBER == 0:
            for z in range(len(param[1])):
                for zz in range(4*len(PROPOTION)):
                    if (np.abs(x - param[1][z][zz][0]) <= CIRCLE_SIZE) and (np.abs(y - param[1][z][zz][1]) <= CIRCLE_SIZE):
                        select_cutpoint = param[1][z][zz]
                        if (np.mod(zz, 4) == 0) or (np.mod(zz, 4) == 1):
                            CONFIRM_POINTS.append(param[1][z][np.mod(zz+2, 4*len(PROPOTION))])
                        else:
                            CONFIRM_POINTS.append(param[1][z][np.mod(zz-2, 4*len(PROPOTION))])
                        if CLICK_NUMBER == 0:
                            CLICK_POINTS.extend([select_cutpoint])
                            CLICK_NUMBER = CLICK_NUMBER + 1
                            
        if CLICK_NUMBER == 1:
            cv2.circle(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), CIRCLE_SIZE, (255, 0, 0), -1)
            for z in range(len(CONFIRM_POINTS)):
                cv2.circle(param[0], (int(CONFIRM_POINTS[z][0]), int(CONFIRM_POINTS[z][1])), CIRCLE_SIZE, (0, 255, 255), -1)
            cv2.imshow('image', param[0])
            for zzz in range(len(CONFIRM_POINTS)):
                if (np.abs(x - CONFIRM_POINTS[zzz][0]) <= CIRCLE_SIZE) and (np.abs(y - CONFIRM_POINTS[zzz][1]) <= CIRCLE_SIZE) :
                    select_cutpoint = CONFIRM_POINTS[zzz]
                    CLICK_POINTS.extend([select_cutpoint])
                    CLICK_NUMBER = CLICK_NUMBER + 1                    
                    break
            if CLICK_NUMBER == 2:
                cv2.line(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), (0, 255, 0), 3)
            cv2.imshow('image', param[0])
                              
def parse(vertexes, image):
    global CLICK_NUMBER, CLICK_POINTS, CONFIRM_POINTS
    possible_points = possible_cutpos(vertexes)
    for j in range(len(possible_points)):
        for k in range(4*len(PROPOTION)):
            cv2.circle(image, (int(possible_points[j][k][0]), int(possible_points[j][k][1])), CIRCLE_SIZE, CIRCLE_COLOR, -1)  
    cv2.imshow('image', image)    
    cv2.setMouseCallback('image', mouse_select, [image, possible_points])
    cv2.waitKey()
#    cut_points = CLICK_POINTS[:]
#    print(cut_points) 


    selected_shape = select_cutshape(vertexes, CLICK_POINTS)
    points_reallocation(cut_points = CLICK_POINTS, vertexes = vertexes, selected_shape = selected_shape)   
    CLICK_NUMBER = 0
    CLICK_POINTS = []
    CONFIRM_POINTS = []
#    vertexes_draw = points_sort(vertexes [draw_order+1])
#    img = draw_polygen(vertexes = np.array(vertexes_draw,'int32'), img = image, color = colors[draw_order+1])
    return vertexes, image
    
def color_select(event, x, y, flag, param):
    global CLICK_COLOR, COLOR_ALL
    if event == cv2.EVENT_LBUTTONDOWN:
        for j in range(len(COLOR_ALL)):
            if (x > 0.9*image_width) and (x < 1.0*image_width) and (y > (0.5+j)*0.1*image_height) and (y < (1.5+j)*0.1*image_height):
                CLICK_COLOR.append(COLOR_ALL[j])
                cv2.line(param, (int(0.9*image_width), int((0.5+j)*0.1*image_height)), (int(1.0*image_width), int((1.5+j)*0.1*image_height)), (40, 40, 40), 2)
                press_image = cv2.imread(image_path+'press1.png')
                param[48:80, 375:547] = press_image                
                break
        cv2.imshow('image', param)
        
def color_fill(event, x, y, flag, param):
    global CLICK_COLOR
    if event == cv2.EVENT_LBUTTONDOWN:
        for j in range(len(param[1])):
            min_x = min(param[1][j][0][0], param[1][j][2][0])
            max_x = max(param[1][j][0][0], param[1][j][2][0])
            min_y = min(param[1][j][0][1], param[1][j][2][1])
            max_y = max(param[1][j][0][1], param[1][j][2][1])
            if (x > min_x) and (x < max_x) and (y > min_y) and (y < max_y):
                cv2.rectangle(param[0], (int(min_x), int(min_y)), (int(max_x), int(max_y)), CLICK_COLOR[0], -1)
                press_image = cv2.imread(image_path+'press1.png')
                param[0][48:80, 375:547] = press_image
                break
        cv2.imshow('image', param[0])
        
        
def fill(vertexes, fill_order, image, image_number):
    global CLICK_COLOR
    CLICK_COLOR = []
    cv2.setMouseCallback('image', color_select, param = image)
    cv2.waitKey()
    image[48:80, 375:547] = 40
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', color_fill, [image, vertexes])
    cv2.waitKey()
    image[48:80, 375:547] = 40
    cv2.imshow('image', image)
    cv2.imwrite(colordata_path+str(IMAGE_ORDER[image_number - 1])+'/'+str(CUT_NUMBER - fill_order + 1)+'.png', image)
    if fill_order == 0:
        cv2.imshow('image', image)
        return image
    return fill(vertexes, fill_order - 1, image, image_number)

def cut(vertexes, cut_order, image, image_number):
    image = np.zeros([image_height, image_width, 3], 'uint8') 
    image[:] = 40
    for i in range(CUT_NUMBER + 1 - cut_order):
        points = np.array(vertexes[i], np.int32)
        cv2.polylines(image, [points], True, (55,255,155), 3)
        cv2.imshow('image', image)
        
    image = parse(vertexes = vertexes, image = image)[1]
    cv2.imwrite(data_path+str(IMAGE_ORDER[image_number - 1])+'/'+str(CUT_NUMBER - cut_order + 1)+'.png', image)
    
    if cut_order == 1:
        image = np.zeros([image_height, image_width, 3], 'uint8')
        image[:] = 40
        for k in range(CUT_NUMBER+1):
            cv2.rectangle(image, (int(vertexes[k][0][0]), int(vertexes[k][0][1])), (int(vertexes[k][2][0]), int(vertexes[k][2][1])), (55,255,155), 3)            
        for j in range(len(COLOR_ALL)):
            cv2.rectangle(image, (int(0.9*image_width), int((0.5+j)*0.1*image_height)), (int(1.0*image_width), int((1.5+j)*0.1*image_height)), COLOR_ALL[j], -1)
        cv2.imshow('image', image)
        
        fill(vertexes = vertexes, fill_order = CUT_NUMBER, image = image, image_number = image_number)
        return image 
    return cut(vertexes, cut_order-1, image, image_number)
        
def iamge_generate(image_number):
    image = np.zeros([image_height, image_width, 3], 'uint8') 
    image[...] = 40
    cv2.namedWindow('image')
    cv2.imshow('image', image)
    cv2.waitKey()
    memory_image = cv2.imread(image_path+str(IMAGE_ORDER[image_number - 1])+'.png')
    resize_image = cv2.resize(memory_image,(int(0.6*image_width), int(0.6*image_height)),interpolation=cv2.INTER_CUBIC)
    image[int(0.2*image_height) : int(0.8*image_height), int(0.2*image_width) : int(0.8*image_width)] = resize_image
    cv2.namedWindow('image')
    cv2.imshow('image', image)
    cv2.waitKey(2000)
    VERTEXES = []     
    vertexes_nocut = [[0.2*image_width, 0.2*image_height], [0.8*image_width, 0.2*image_height], [0.8*image_width, 0.8*image_height], [0.2*image_width, 0.8*image_height]]
    VERTEXES.append(vertexes_nocut)
    image = cut(vertexes = VERTEXES, cut_order = CUT_NUMBER, image = image, image_number = image_number)   
    cv2.imshow('image', image)
    
    if image_number == 1:
        print('done')
        return    
    return iamge_generate(image_number-1)
    
    
def main():
    np.random.shuffle(IMAGE_ORDER)
    iamge_generate(image_number = IMAGE_NUMBER)

if __name__ == '__main__':
    main()
    
    
        
    