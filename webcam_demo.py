import tensorflow as tf
import cv2
import time
import argparse
import math
import posenet
import numpy as np

from tkinter import *
from tkinter import colorchooser

#used to calculate the angles
def find_angle(a, b, c):
    try:
        ang = int(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])))
        return ang + 360 if ang < 0 else ang
    except Exception:
        return 0







URL = 'http://10.42.0.211:8080/video'
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=2)
parser.add_argument('--vid_url')
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.35)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--pose_num', type=int, default=1)
parser.add_argument('--color_choice', type=int, default=0)
args = parser.parse_args()



#selecting 1st color from GUI for sticker
clr=[]
clr2=[]




def call_me():
    global clr
    clr=colorchooser.askcolor(title="select 1st color")

def call_me2():
    global clr2
    clr2=colorchooser.askcolor(title="select 2nd color")

if args.color_choice==1:

    #selecting color from gui
    root=Tk()
    button=Button(root,text="Change 1st color",command=call_me)
    button.pack()
    root.geometry("300x300")
    root.mainloop()

    clr=list(clr[0])
    clr.reverse()
    clr=np.uint8([[clr]])
    print("color1= ",clr)
    hsv_clr = cv2.cvtColor(clr,cv2.COLOR_BGR2HSV)
    print("hsv_color1= ",hsv_clr)

    #selecting 2nd color from GUI for sticker
    root=Tk()
    button=Button(root,text="Change 2nd color",command=call_me2)
    button.pack()
    root.geometry("300x300")
    root.mainloop()

    clr2=list(clr2[0])
    clr2.reverse()
    clr2=np.uint8([[clr2]])
    print("color2= ",clr2)
    hsv_clr2 = cv2.cvtColor(clr2,cv2.COLOR_BGR2HSV)
    print("hsv_color2= ",hsv_clr2)

else:
    clr=[166,27,240]
    clr=np.uint8([[clr]])
    hsv_clr=cv2.cvtColor(clr,cv2.COLOR_BGR2HSV)
    clr2=[52,238,217]
    clr2=np.uint8([[clr2]])
    hsv_clr2=cv2.cvtColor(clr2,cv2.COLOR_BGR2HSV)




def main():
    new1=[]
    new_adj=[]
    
    print("HSV-COLOR main= ",hsv_clr)
    print("COLOR main= ",clr)
    print("HSV-COLOR2 main2= ",hsv_clr2)
    print("COLOR main2= ",clr2)




    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        #print('output stride = ',output_stride)
        #output_stride= 8

        if args.cam_id == -1:
            cap = cv2.VideoCapture(args.vid_url)
        elif args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        start = time.time()
    
        if args.cam_id == 0:
            window_name = 'FRONT VIEW'
        else:
            window_name = 'SIDE VIEW'
        frame_count = 0
        body_rotation = {}


        


        while True:
            body_coord = {}
            for nodal_points in posenet.PART_NAMES:
                body_coord[nodal_points] = [0, 0]

            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=args.pose_num,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image,new1,new_adj,pts_new= posenet.draw_skel_and_kp(display_image, pose_scores, keypoint_scores, keypoint_coords, body_rotation,frame_count,new1,new_adj,min_pose_score=0.15, min_part_score=0.1)
            



            #detect color of user choice
            hsv=cv2.cvtColor(overlay_image,cv2.COLOR_BGR2HSV)

            if ((int)(clr2[0][0][0]) + (int)(clr2[0][0][1]) + (int)(clr2[0][0][2]) <= 30):
                print("in black2")
                green_lower=np.array([0,0,0],np.uint8)
                green_upper=np.array([20,0,0],np.uint8)
            elif(hsv_clr2[0][0][0] >=10):
                green_lower=np.array([hsv_clr2[0][0][0]-10,100,100],np.uint8)
                green_upper=np.array([hsv_clr2[0][0][0]+10,255,255],np.uint8)
            else:
                green_lower=np.array([hsv_clr2[0][0][0],100,100],np.uint8)
                green_upper=np.array([hsv_clr2[0][0][0]+10,255,255],np.uint8)






            if ((int)(clr[0][0][0]) + (int)(clr[0][0][1]) + (int)(clr[0][0][2]) <= 30):
                print("in black")
                blue_lower=np.array([0,0,0],np.uint8)
                blue_upper=np.array([20,0,0],np.uint8)

            elif(hsv_clr[0][0][0] >=10):
                blue_lower=np.array([hsv_clr[0][0][0]-10,100,100],np.uint8)
                blue_upper=np.array([hsv_clr[0][0][0]+10,255,255],np.uint8)
            else:
                blue_lower=np.array([hsv_clr[0][0][0],100,100],np.uint8)
                blue_upper=np.array([hsv_clr[0][0][0]+10,255,255],np.uint8)

            #finding the range of color in the image
            blue=cv2.inRange(hsv,blue_lower,blue_upper)
            green=cv2.inRange(hsv,green_lower,green_upper)

            #morphological transformation,Dillation
            kernal_blue=np.ones((5,5),"uint8")
            kernal_green=np.ones((5,5),"uint8")

            blue=cv2.dilate(blue,kernal_blue)
            res_blue=cv2.bitwise_and(overlay_image,overlay_image,mask=blue)

            green=cv2.dilate(green,kernal_green)
            res_green=cv2.bitwise_and(overlay_image,overlay_image,mask=green)
            #tracking the 1st color
            (contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for pic,contour in enumerate(contours):
                area=cv2.contourArea(contour)
                if area>200:
                    x,y,w,h=cv2.boundingRect(contour)
                    #overlay_image=cv2.rectangle(overlay_image,(x,y),(x+w,y+h),(0,255,0),2)
                    cen_x1=(int)(((2*x)+w)/2)
                    cen_y1=(int)(((2*y)+h)/2)
                    cen_1=(cen_x1,cen_y1)
                    overlay_image=cv2.circle(overlay_image,cen_1,2,(255,0,0),2)

                    #drawing for sticker
                    try:
                        overlay_image=cv2.line(overlay_image,(pts_new[15][0],pts_new[15][1]),cen_1,(255,255,255),4)
                        ang1=find_angle(pts_new[13],pts_new[15],cen_1)
                        if ang1>180:

                            ang1=360-ang1
                        cv2.putText(overlay_image,"{}".format(ang1), cen_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
                    except Exception:
                        print('not found')


            #Tracking 2nd color
            (contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic,contour in enumerate(contours):
                area=cv2.contourArea(contour)
                if area>300:
                    x,y,w,h=cv2.boundingRect(contour)
                    #overlay_image=cv2.rectangle(overlay_image,(x,y),(x+w,y+h),(0,255,0),2)
                    cen_x2=(int)(((2*x)+w)/2)
                    cen_y2=(int)(((2*y)+h)/2)
                    cen_2=(cen_x2,cen_y2)
                    overlay_image=cv2.circle(overlay_image,cen_2,2,(255,0,0),2)

                    #drawing for sticker
                    try:
                        overlay_image=cv2.line(overlay_image,(pts_new[16][0],pts_new[16][1]),cen_2,(255,255,0),4)
                        ang1=find_angle(pts_new[14],pts_new[16],cen_2)
                        if ang1>180:
                            ang1=360-ang1
                        cv2.putText(overlay_image,"{}".format(ang1), cen_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
                        

                    except Exception:
                        print('not found')





            cv2.imshow("masked_color1",res_blue)
            cv2.imshow("masked_color2",res_green)
            cv2.imshow(window_name, overlay_image)
            frame_count += 1
            #start1=time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        print('Average FPS: ', frame_count / (time.time() - start))






if __name__ == "__main__":
    main()
