import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
left_ear_cascade_path="C:/Users/rokas/opencv-master/opencv-master/data/haarcascades/haarcascade_mcs_rightear.xml"
#カスケードファイルの読み込み
ear_cascade=cv2.CascadeClassifier(left_ear_cascade_path)

while(True):
    ret, frame = cap.read()
    
    #グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #平滑化
    gray_smooth = cv2.GaussianBlur(gray, (11,11), 0)
    #二値化
    ret2, thresh = cv2.threshold(gray_smooth, 127,255,cv2.THRESH_BINARY)
    #輪郭抽出
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    
    print("contour[1] length = ", len(contours[1]))
    approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    
    for e in contours:
        cv2.approxPolyDP(e, 0.01*cv2.arcLength(e, true), true)
        if(e.ContourArea() > 2000.0):    
            #輪郭構成座標の保存
            with open('face_contour.txt', 'w') as f:
                for x in contours[0]:
                    f.write("contour:"+str(x) + "/n")
            

    #デバッグ用に座標表示
    print(contours[0])
    print("一巡終了")

    #if cv2.contourArea(approx) > 2000:
    #輪郭描画
    frame = cv2.drawContours(frame, contours[0], -1, (0, 0, 255), 5)
    #key = cv2.waitKey(33)

    #カスケードファイルから右耳を含む長方形の頂点4点を求める
    ears = ear_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(30,30))
    print("ears : ", ears)
    f = open('face_contour.txt', 'a')
    for x in ears:
        f.writelines("ear:"+str(x))
    f.close()

    #検知した矩形4点の座標をもとに長方形描画
    if 0 != len(ears):
        BORDER_COLOR = (255, 255, 255) # 線色を白に
        for rect in ears:
            # 顔検出した部分に枠を描画
            cv2.rectangle(
                image,
                tuple(rect[0:2]),
                tuple(rect[0:2] + rect[2:4]),
                BORDER_COLOR,
                thickness=2
            )
    
    #更新後の座標表示
    cv2.imshow("frame", frame)

    #escで終了
    if cv2.waitKey(33) == 27:
        break

#リソース解放
cap.release()
cv2.destroyAllWindows()