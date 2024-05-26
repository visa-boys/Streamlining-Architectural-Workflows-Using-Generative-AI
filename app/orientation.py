import math
import pickle
def center_of_plot(boxes):
    # Calculate the minimum and maximum x and y coordinates
    x_min = min(min(box[0], box[2]) for box in boxes)
    x_max = max(max(box[0], box[2]) for box in boxes)
    y_min = min(min(box[1], box[3]) for box in boxes)
    y_max = max(max(box[1], box[3]) for box in boxes)

    # Calculate the center of the floorplan
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return (center_x,center_y)
def ret_comb():
    with open('rectangles.pkl', 'rb') as f:
        rectangles = pickle.load(f)
    #rectangles = [(64, 89, 143, 163), (59, 19, 145, 89), (60, 18, 86, 53), (84, 3, 146, 19), (143, 89, 201, 143), (142, 142, 183, 163), (35, 87, 63, 116)]
    rectangles = [[x, 255 - y1, x2, 255 - y2] for x, y1, x2, y2 in rectangles]
    (center_x,center_y)=center_of_plot(rectangles)
    orientation=[]
    #vastu=[]
    #result=[]
    #index=0
    for i in range(len(rectangles)):
        # Calculate centroid
        x0,y0,x1,y1=rectangles[i]
        y_centroid = (y0 + y1) / 2
        x_centroid = (x0 + x1) / 2
        #print(x_centroid,y_centroid)

        deltaX = x_centroid - center_x
        deltaY = y_centroid - center_y

        degrees_temp = math.atan2(deltaX, deltaY)/math.pi*180

        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp
        
        compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        compass_lookup = round(degrees_final / 45)
        orientation.append(compass_lookup)
        #orientation.append(compass_brackets[compass_lookup])
        #vastu.append(vastu_scores[c][compass_lookup])
        #room_types=combinations[index][0]
        #result.append([room_types[i],compass_lookup])
    return orientation