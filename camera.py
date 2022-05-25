######################################

import requests
import cv2 as cv
import face_recognition as fr
import datetime
import pickle

#########Process Image##############

def process_image(frame_rgb, Name_Student,Encodings_Student):
	global student
	frame_rgb = cv.resize(frame_rgb, (0,0), fx=scaleX, fy=scaleY)
	face_positions = fr.face_locations(frame_rgb)
	all_face_encodings = fr.face_encodings(frame_rgb,face_positions)

	for(top,right,bottom,left), face_encoding in zip(face_positions, all_face_encodings):
		student_name = 'Not Recognised'
		matches = fr.compare_faces(Encodings_Student,face_encoding, tolerance=0.7)
		if True in matches:
			first_match_index = matches.index(True)
			#color = (0,255,255)
			student_name = Name_Student[first_match_index]
			if student_name not in student:
				requests.post(base_url+student_name+'\n'+str(datetime.datetime.now()))	
				student.append(student_name)
		#else:
			#color = (0,0,255)

		#cv.rectangle(frame_rgb,(right,top),(left,bottom),color,2)
		#return frame_rgb

		

#######Initialize Video Device#######

cam= cv.VideoCapture(0)
font= cv.FONT_HERSHEY_SIMPLEX
scaleX = 0.25
scaleY = 0.25

##########Telegram API Key###########
bot_token = '5307156877:AAFj1Yr8Msxn1jyXBEsrOwLDQ1TCHMFjOm4'
bot_chatID = '630184663'
base_url = 'https://api.telegram.org/bot'+bot_token+'/sendMessage?chat_id=-'+bot_chatID+'&text='

##################################### 
#requests.get(base_url+str(100))

student = []

######Load Training Pickle File######

with open('student.pkl','rb') as stu:
	Name_Student = pickle.load(stu)
	Encodings_Student = pickle.load(stu)

#####################################

while True:
	ret,frame = cam.read()
	if not ret:
		break

	frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
	process_image(frame_rgb,Name_Student,Encodings_Student)
	
	#frame = cv.cvtColor(frame_rgb,cv.COLOR_RGB2BGR)
	cv.imshow('Cam1',frame)
	if cv.waitKey(1)==ord('q'):
		break

#####################################

cam.release()
cv.destroyAllWindows()

#####################################
