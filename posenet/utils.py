import cv2
import numpy as np
import copy
import posenet.constants
import math


#used to calculate the angles
def find_angle(a, b, c):
    try:
        ang = int(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])))
        return ang + 360 if ang < 0 else ang
    except Exception:
        return 0
    



def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img

def draw_skel_and_kp(img, instance_scores, keypoint_scores, keypoint_coords, body_rotation,frame_count,new1,new_adj,min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []

    

    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    #converting into integer
    pts = cv2.KeyPoint_convert(cv_keypoints)
    print('frame-count(utils)=',frame_count)
    frame_count=frame_count+1
    
    pts_ret=cv2.KeyPoint_convert(new1)

    #new1 = copy.copy(cv_keypoints) if len(new1) <= 0 else new1

    if(frame_count %15 ==0 or frame_count==1):
        #stabilazation of frames (pranoy)
        out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.polylines(out_img, adjacent_keypoints, False, (255, 255, 255), 4)

        new_adj=copy.copy(adjacent_keypoints)
        new1=copy.copy(cv_keypoints)
        


        
    else:
        
        out_img = cv2.drawKeypoints(out_img, new1, outImage=np.array([]), color=(255, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.polylines(out_img, new_adj, False, (255, 255, 255), 4)

        pts_new=cv2.KeyPoint_convert(new1)

        


        #angle calculation (pranoy)
        print('pts-new= ',pts_new)
        try:
            ang1=find_angle(pts_new[5],pts_new[7],pts_new[9])
            
            if ang1>180:
                ang1=360-ang1
            cv2.putText(out_img,"{}".format(ang1), (pts[7][0],pts[7][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
            #printing on the right side of the window.
            #text1= 'Left Elbow= '+str(ang1)
            #cv2.putText(out_img,text1, (400,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (222, 0, 0), 1,lineType=cv2.LINE_AA)



            ang2=find_angle(pts_new[7],pts_new[5],pts_new[11])
            cv2.putText(out_img,"{}".format(ang2), (pts[5][0],pts[5][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 2,lineType=cv2.LINE_AA)
            #printing on the right side of the window.
            #text2= 'Left Shoulder= '+str(ang2)
            #cv2.putText(out_img,text2, (400,37), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (222, 0, 0), 1,lineType=cv2.LINE_AA)




            ang3=find_angle(pts_new[10],pts_new[8],pts_new[6])
            if ang3>180:
                ang3=360-ang3
            cv2.putText(out_img,"{}".format(ang3), (pts[8][0],pts[8][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)         
            #printing on the right side of the window.
            #text3='Right Elbow= '+str(ang3)
            #cv2.putText(out_img,text3, (400,98), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 222, 0), 1,lineType=cv2.LINE_AA)






            ang4=find_angle(pts_new[12],pts_new[6],pts_new[8])
            cv2.putText(out_img,"{}".format(ang4), (pts[6][0],pts[6][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
            #printing on the right side of the window.
            #text4='Right shoulder='+str(ang4)
            #cv2.putText(out_img,text4, (400,115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 222, 0), 1,lineType=cv2.LINE_AA)


            ang5=find_angle(pts_new[15],pts_new[13],pts_new[11])
            cv2.putText(out_img,"{}".format(ang5), (pts[13][0],pts[13][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)                  
            #printing on the right side
            #text5='Left Knee=' + str(ang5)
            #cv2.putText(out_img,text5, (400,71), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (222, 0, 0), 1,lineType=cv2.LINE_AA)





            ang6=find_angle(pts_new[12],pts_new[14],pts_new[16])
            cv2.putText(out_img,"{}".format(ang6), (pts[14][0],pts[14][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)         
            #printing on the right side
            #text6='Right Knee='+str(ang6)
            #cv2.putText(out_img,text6, (400,149), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 222, 0), 1,lineType=cv2.LINE_AA)


            ang7=find_angle(pts_new[5],pts_new[11],pts_new[13])
            cv2.putText(out_img,"{}".format(ang7), (pts[11][0],pts[11][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
            #printing on the right side
            #text7='left heap= '+ str(ang7)
            #cv2.putText(out_img,text7, (400,54), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (222, 0, 0), 1,lineType=cv2.LINE_AA)           





            ang8=find_angle(pts_new[6],pts_new[12],pts_new[14])
            cv2.putText(out_img,"{}".format(ang8), (pts[12][0],pts[12][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,lineType=cv2.LINE_AA)
            #printing on the right side
            #text8='Right Heap='+str(ang8)
            #cv2.putText(out_img,text8, (400,132), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 222, 0), 1,lineType=cv2.LINE_AA)    

        #Drawing using sticker
    





        except Exception:
            print('points not found')







    
    


    return out_img,new1,new_adj,pts_ret

