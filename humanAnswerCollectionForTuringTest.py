from __future__ import division
import numpy as np
import os
import cv2
import itertools as it

project_path = "redraw/"
data_path = os.path.join(project_path, "data/")
image_path = os.path.join(project_path, "images/")


image_width = 720
image_height = 720
BLANK_PROPORTION = 0.1
ORIGIN_IMGWIDTH = 960
ORIGIN_IMGHEIGHT = 960
CUT_NUMBER = 3
IMAGE_NUMBER = 48
IMAGE_ORDER = [0, 3, 5, 6, 8, 14, 17, 20, 22, 23, 30, 31, 32, 35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 49, 51, 53, 54, 57, 59, 60, 63, 69, 112, 123, 130, 136, 147, 160, 171, 176, 200, 207, 208, 213, 220, 223, 235, 239]
COLOR_ALL = [[0,0,0],[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
COLOR_DATA = []
VERTEXES_DATA = []
CIRCLE_SIZE = 4
LINE_SIZE = 7
CIRCLE_COLOR = [0,0,255]
INTERVAL = 240 * image_width / ORIGIN_IMGWIDTH * (1 - BLANK_PROPORTION)
VERTEXES_PERPOLYGEN = 4
CLICK_NUMBER = 0
CLICK_POINTS = []
CONFIRM_POINTS = []
CLICK_COLOR = []
SUB_NUM = raw_input('subject_num:')

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
        if len_xside > INTERVAL:
            for k in range(int(len_xside//INTERVAL) - 1):
                possible_cutpoints.append([min_x + (k+1) * INTERVAL, min_y])
                possible_cutpoints.append([min_x + (k+1) * INTERVAL, max_y])     
        if len_yside > INTERVAL:
            for k in range(int(len_yside//INTERVAL) - 1):
                possible_cutpoints.append([min_x, min_y + (k+1) * INTERVAL])
                possible_cutpoints.append([max_x, min_y + (k+1) * INTERVAL])
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
                for zz in range(len(param[1][z])):
                    if (np.abs(x - param[1][z][zz][0]) <= CIRCLE_SIZE+10) and (np.abs(y - param[1][z][zz][1]) <= CIRCLE_SIZE+10):
                        select_cutpoint = param[1][z][zz]
                        if (np.mod(zz, 2) == 0):
                            CONFIRM_POINTS.append(param[1][z][zz+1])
                        else:
                            CONFIRM_POINTS.append(param[1][z][zz-1])
                        if CLICK_NUMBER == 0:
                            CLICK_POINTS.extend([select_cutpoint])
                            CLICK_NUMBER = CLICK_NUMBER + 1
                            
        if CLICK_NUMBER == 1:
            cv2.circle(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), CIRCLE_SIZE, (255, 0, 0), -1)
            for z in range(len(CONFIRM_POINTS)):
                cv2.circle(param[0], (int(CONFIRM_POINTS[z][0]), int(CONFIRM_POINTS[z][1])), CIRCLE_SIZE, (0, 128, 0), -1)
            cv2.imshow('image', param[0])
            for zzz in range(len(CONFIRM_POINTS)):
                if (np.abs(x - CONFIRM_POINTS[zzz][0]) <= CIRCLE_SIZE+10) and (np.abs(y - CONFIRM_POINTS[zzz][1]) <= CIRCLE_SIZE+10) :
                    select_cutpoint = CONFIRM_POINTS[zzz]
                    CLICK_POINTS.extend([select_cutpoint])
                    CLICK_NUMBER = CLICK_NUMBER + 1                    
                    break
            if CLICK_NUMBER == 2:
                cv2.line(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), (255, 255, 255), LINE_SIZE)
            cv2.imshow('image', param[0])
                              
def parse(vertexes, image, cut_order):
    global CLICK_NUMBER, CLICK_POINTS, CONFIRM_POINTS
    possible_points = possible_cutpos(vertexes)
    for j in range(len(possible_points)):
        for k in range(len(possible_points[j])):
            cv2.circle(image, (int(possible_points[j][k][0]), int(possible_points[j][k][1])), CIRCLE_SIZE, CIRCLE_COLOR, -1)  
    cv2.imshow('image', image)
    parsekey = 100000
    while len(CLICK_POINTS) != 2 and parsekey != 13:
        cv2.setMouseCallback('image', mouse_select, [image, possible_points])
        parsekey = cv2.waitKey()
    
    if parsekey == 13:
        cut_order = 1
    else:
        selected_shape = select_cutshape(vertexes, CLICK_POINTS)
        points_reallocation(cut_points = CLICK_POINTS, vertexes = vertexes, selected_shape = selected_shape)   

    CLICK_NUMBER = 0
    CLICK_POINTS = []
    CONFIRM_POINTS = []
    return vertexes, image, cut_order
    
def cut(vertexes, cut_order, image, image_number):
    global SUB_NUM
    for i in range(CUT_NUMBER + 1 - cut_order):
        points = np.array(vertexes[i], np.int32)
        cv2.polylines(image, [points], True, (255, 255, 255), LINE_SIZE)
        cv2.imshow('image', image)
    cv2.imwrite(data_path+'demo'+str(IMAGE_ORDER[image_number - 1])+'_'+SUB_NUM+'cut'+ str(CUT_NUMBER - cut_order + 1)+'.png', image)
    image, cut_order = parse(vertexes = vertexes, image = image, cut_order = cut_order)[1:3]

    if cut_order == 1:
        for k in range(len(vertexes)):
            cv2.rectangle(image, (int(vertexes[k][0][0]), int(vertexes[k][0][1])), (int(vertexes[k][2][0]), int(vertexes[k][2][1])), (255, 255, 255), LINE_SIZE)            
        cv2.imshow('image', image)
        cv2.imwrite(data_path+'demo'+str(IMAGE_ORDER[image_number - 1])+'_'+SUB_NUM+'humanAnswer.png', image)
        return image 
    return cut(vertexes, cut_order-1, image, image_number)
        
def iamge_generate(image_number):
    image = np.zeros([image_height, image_width, 3], 'uint8') 
    blank_proportion = BLANK_PROPORTION
    memory_image = cv2.imread(image_path+'demo'+str(IMAGE_ORDER[image_number - 1])+'.png')
    resize_image = cv2.resize(memory_image,(int((1-blank_proportion)*image_width), int((1-blank_proportion)*image_height)),interpolation=cv2.INTER_CUBIC)
    image[int((blank_proportion/2)*image_height) : int((1-blank_proportion/2)*image_height), int((blank_proportion/2)*image_width) : int((1-blank_proportion/2)*image_width)] = resize_image
    cv2.namedWindow('image')
    cv2.imshow('image', image)
    VERTEXES = []     
    vertexes_nocut = [[(blank_proportion/2)*image_width, (blank_proportion/2)*image_height], [(1-blank_proportion/2)*image_width, (blank_proportion/2)*image_height], [(1-blank_proportion/2)*image_width, (1-blank_proportion/2)*image_height], [(blank_proportion/2)*image_width, (1-blank_proportion/2)*image_height]]
    VERTEXES.append(vertexes_nocut)
    image = cut(vertexes = VERTEXES, cut_order = CUT_NUMBER, image = image, image_number = image_number)   
    cv2.imshow('image', image)
    print(image_number)
    key = cv2.waitKey()
    if key == 27:
        image_number = 1
    if image_number == 1:
        print('done')
        cv2.destroyAllWindows()
        return    
    return iamge_generate(image_number-1)
    
    
def main():
    iamge_generate(image_number = IMAGE_NUMBER)

if __name__ == '__main__':
    main()
    
    
        
    
