# 尚缺:
# 偵測到拍照完音樂停&關閉拍照框
# 燈暗與拍照功能切割
# 連資料庫&寄e-mail

# 目前功能:
# 設定時間到鬧鐘鈴聲開啟&鏡頭開始偵測

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
import re
import time
import requests
from datetime import datetime, timedelta
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
#delete package
import shutil
#connect to mysql
import mysql.connector
import MySQLdb
maxdb = mysql.connector.connect(
  host = "192.168.36.28",
  port=3306,
  user = "root",
  password = "10219873",
  database = "demo"
  )
cursor=maxdb.cursor()
emergency_users__result_row=()
cursor.execute("SELECT * FROM emergency_users")
emergency_users__result = cursor.fetchall()
emergency_users__result_row=emergency_users__result[0]
mail=emergency_users__result_row[1]
#From emergency_users

row=()
cursor=maxdb.cursor()
cursor.execute("SELECT * FROM users")
result = cursor.fetchall()
users_row=result[0]
users_name=users_row[1]
print(users_name)
users_medicine_1=users_row[4]
users_medicine_2=users_row[5]
users_medicine_3=users_row[6]
users_medicine_4=users_row[7]
users_medicine_5=users_row[8]

# LED initialize
import RPi.GPIO as GPIO	  #引入 GPIO 套件
LED1_PIN = 18  #控制 LED 的腳位，實體編號 11 的腳位，其 BCM 編號為 17。
LED2_PIN = 27  #控制 LED 的腳位，實體編號 13 的腳位，其 BCM 編號為 27。
LED3_PIN = 22  #控制 LED 的腳位，實體編號 15 的腳位，其 BCM 編號為 22。
GPIO.setmode(GPIO.BCM)	 ##使用 BCM 編號方式。 
GPIO.setup(LED1_PIN, GPIO.OUT)	 #腳位設定為輸出模式。
GPIO.setup(LED2_PIN, GPIO.OUT)	 #腳位設定為輸出模式。
GPIO.setup(LED3_PIN, GPIO.OUT)	 #腳位設定為輸出模式。

# from right now to the order time	 
def get_seconds(h, m, s):
	"""获取当前时间与程序启动时间间隔秒数"""

	# 设置程序启动的时分秒
	time_pre = '%s:%s:%s' % (h, m, s)
	# 获取当前时间
	time1 = datetime.now()
	# 获取程序今天启动的时间的字符串格式
	time2 = time1.date().strftime('%Y-%m-%d') + ' ' + time_pre
	# 转换为datetime格式
	time2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')
	# 判断当前时间是否晚于程序今天启动时间，若晚于则程序启动时间增加一天
	if time1 > time2:
		time2 = time2 + timedelta(days=1)

	return time.mktime(time2.timetuple()) - time.mktime(time1.timetuple())

def start_camera():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--cascade", required=True,
		help = "path to where the face cascade resides")
	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-o", "--output", required=True,
		help="path to output directory")	
	args = vars(ap.parse_args())

	# load the known faces and embeddings along with OpenCV's Haar
	# cascade for face detection
	print("[INFO] loading encodings + face detector...")
	data = pickle.loads(open(args["encodings"], "rb").read())
	detector = cv2.CascadeClassifier(args["cascade"])

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	vs = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)
	total = 0

	# start the FPS counter
	fps = FPS().start()

	flag = True
	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to 500px (to speedup processing)
		frame = vs.read()
		orig = frame.copy()
		frame = imutils.resize(frame, width=500)
		
		# convert the input frame from (1) BGR to grayscale (for face
		# detection) and (2) from BGR to RGB (for face recognition)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# detect faces in the grayscale frame
		rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		# OpenCV returns bounding box coordinates in (x, y, w, h) order
		# but we need them in (top, right, bottom, left) order, so we
		# need to do a bit of reordering
		boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

		# compute the facial embeddings for each face bounding box
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		# loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding)
			name = "Unknown"

			# check to see if we have found a match
			if True in matches:
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				
				# loop over the matched indexes and maintain a count for
				# each recognized face face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
				name = max(counts, key=counts.get)			
				
			# update the list of names
			names.append(name)

		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
			# draw the predicted face name on the image
			cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)
			# LED turnon
			if flag and name==users_name:
				GPIO.output(LED1_PIN, GPIO.HIGH)
				GPIO.output(LED2_PIN, GPIO.HIGH)
				GPIO.output(LED3_PIN, GPIO.HIGH)
				flag = False
				# take picture
				p = os.path.sep.join([args["output"], "{}.png".format(
				str(total).zfill(5))])
				cv2.imwrite(p, orig)
				total += 1
				print("picture has been taken!")
				#send email
				if __name__ == '__main__':
					fromaddr = mail
					password = '10219873'
					toaddrs = [mail]
					content = '已經吃藥囉'
					textApart = MIMEText(content)
					imageFile = 'morris/00000.png'
					imageApart = MIMEImage(open(imageFile, 'rb').read(), imageFile.split('.')[-1])
					imageApart.add_header('Content-Disposition', 'attachment', filename=imageFile)
					m = MIMEMultipart()
					m.attach(textApart)
					m.attach(imageApart)
					m['Subject'] = 'title'
					try:
						server = smtplib.SMTP('smtp.gmail.com', 587)
						server.ehlo()  # 向Gamil发送SMTP 'ehlo' 命令
						server.starttls()
						server.login(fromaddr,password)
						server.sendmail(fromaddr, toaddrs, m.as_string())
						print('success')
						server.quit()
					except smtplib.SMTPException as e:
						print('error:',e) #打印错误
		
		# display the image to our screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	#delete all pictures
	shutil.rmtree('morris')
	os.mkdir('morris')
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

	# LED turnoff
	GPIO.output(LED1_PIN, GPIO.LOW)
	GPIO.output(LED2_PIN, GPIO.LOW)
	GPIO.output(LED3_PIN, GPIO.LOW)
	GPIO.cleanup()	#確保程式結束後，用到的腳位皆回復到預設狀態。

def T1_job():
	start_camera()

def T2_job():
	# 获取音乐文件绝对地址
	mp3path2 = 'piggg.mp3'
	# 先播放一首音乐做闹钟
	os.system('mplayer %s' % mp3path2)

# 設定鬧鐘時間(hhmmss)	
timelist=['134000', '135720', '135100']
# 現在時間
now = datetime.now()
# 時間格式擷取(hhmmss)
midtime = now.strftime('%H%M%S')
timelist.append(midtime)
print(timelist)
timelist.sort()
print(timelist)
index = timelist.index(midtime)
print(index, timelist[index])
if (index+1)==len(timelist):
	index = 0
	print('next', index, timelist[index])
# 取最接近現在時間者為鬧鐘時間
h = timelist[index+1][:2]
m = timelist[index+1][2:4]
s = timelist[index+1][4:]
timelist.remove(midtime)
print(timelist)
# 等時間到再開始執行
sleeptime = get_seconds(h, m, s)
print(sleeptime)
time.sleep(sleeptime)
# thread一個是偵測、另一個是放音樂
thread_1 = threading.Thread(target=T1_job, name='T1')
thread_2 = threading.Thread(target=T2_job, name='T2')
thread_1.start() # 开启T1
thread_2.start() # 开启T2
print("all done\n")
