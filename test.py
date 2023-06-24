from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pywhatkit

student_number = "+919740736439"
faculty = "+919380393363"
list_marked=[]    


def what_app(number, msg):
    #pywhatkit.sendwhatmsg_instantly(number, msg)
    print("message whatsapp")
 
video=cv2.VideoCapture(1
                       )
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
    

with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)
    
    
present_students = {}
required_duration = 1 * 60 



# List all student
All_students = list(set(LABELS))     

def mark_attendance(student_id):
    if student_id in list_marked:
        pass
    else:
        list_marked.append(student_id)
        print(f"Attendance marked for student ID: {student_id}")
        
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
       
        msg = f"attendence marked ID: {student_id}"

        what_app(student_number, msg)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(FACES, LABELS)



COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    frame = cv2.flip(frame,1)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        
        if str(output[0]) in present_students:
            present_students[str(output[0])] += 1
        else:
            present_students[str(output[0])] = 1
            
        if present_students[str(output[0])] >= required_duration:
            # Mark attendance for the student
            mark_attendance(str(output[0]))
            
            
            
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance=[str(output[0]), str(timestamp)]
        
    
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    
    
        
    if k==ord('q'):
  
        video.release()
        cv2.destroyAllWindows()
        
        
        All_students = set(All_students)
        #present_sorted = list_marked
        list_marked = set(list_marked)
        
        absent_students = All_students - list_marked
        #absent_sorted = list(absent_students) 
        
        print("Messaged Faculty!")
        day_name= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
        day_index = datetime.fromtimestamp(ts).strptime(date,"%d-%m-%Y").weekday()
        
                         
        #pywhatkit.sendwhatmsg_instantly(faculty, f"Today ({day_name[day_index]} {date}) \nToday present: {len(list_marked)}/{len(All_students)}\nPresent : {present_sorted.sort()}\nAbsentee : {absent_sorted.sort()}")
        msg = f"Today ({day_name[day_index]} {date}) \nToday present: {len(list_marked)}/{len(All_students)}\nPresent : {list_marked}\nAbsentee : {absent_students}"
        what_app(faculty, msg)

        break
#video.release()
#cv2.destroyAllWindows()

