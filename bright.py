import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=2)
mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume_ctrl=cast(interface,POINTER(IAudioEndpointVolume))

SMOOTH=0.2
prev_brightness=sbc.get_brightness(display=0)[0]
prev_volume=volume_ctrl.GetMasterVolumeLevelScalar()

def smooth(prev,current):
    return prev+SMOOTH*(current-prev)

while True:
    ret,frame=cap.read()
    if not ret:break
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    if result.multi_hand_landmarks:
        for lm,handness in zip(result.multi_hand_landmarks,result.multi_handedness):
            label=handness.classification[0].label
            mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)
            h,w,_=frame.shape
            thumb=lm.landmark[4]
            index=lm.landmark[8]
            x1,y1=int(thumb.x*w),int(thumb.y*h)
            x2,y2=int(index.x*w),int(index.y*h)
            cv2.circle(frame,(x1,y1),10,(255,0,255),cv2.FILLED)
            cv2.circle(frame,(x2,y2),10,(255,0,255),cv2.FILLED)
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
            distance=np.hypot(x2-x1,y2-y1)
            if label=="Right":
                target=np.interp(distance,[20,200],[0,100])
                prev_brightness=smooth(prev_brightness,target)
                bright=int(np.clip(prev_brightness,0,100))
                sbc.set_brightness(bright)
                cv2.rectangle(frame,(50,150),(85,400),(0,255,0),3)
                bar=np.interp(bright,[0,100],[400,150])
                cv2.rectangle(frame,(50,int(bar)),(85,400),(0,255,0),cv2.FILLED)
                cv2.putText(frame,f'Brightness:{bright}%',(40,430),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            elif label=="Left":
                target=np.interp(distance,[20,200],[0,1])
                prev_volume=smooth(prev_volume,target)
                vol=float(np.clip(prev_volume,0,1))
                volume_ctrl.SetMasterVolumeLevelScalar(vol,None)
                cv2.rectangle(frame,(550,150),(585,400),(255,0,0),3)
                bar=np.interp(vol,[0,1],[400,150])
                cv2.rectangle(frame,(550,int(bar)),(585,400),(255,0,0),cv2.FILLED)
                cv2.putText(frame,f'Volume:{int(vol*100)}%',(500,430),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("Brightness&Volume",frame)
    if cv2.waitKey(1)&0xFF==27:break
cap.release()
cv2.destroyAllWindows()
