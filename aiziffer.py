#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright (C) 2022 - Markus Barth



"""

import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import socket
import struct
import sched, time
import sys
import cv2
import math
import numpy as np
from configparser import ConfigParser
import tflite_runtime.interpreter as tflite
import paho.mqtt.publish as publish
import urllib.request


def rotate_image(image, angle):
    image_center = tuple (np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D (image_center, angle, 1.0)
    result = cv2.warpAffine (image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def my_namer(default_name):
    # This will be called when doing the log rotation
    # default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    # Do any manipulations to that name here, for example this changes the name to Rotate_Test.YYYY-MM-DD.txt
    base_filename, ext, date = default_name.split(".")
    return f"{base_filename}.{date}.{ext}"


config = ConfigParser()
config.read('wasser/config.ini')

LogFile                 = config.getboolean ('Debug', 'Logfile')
LogfileRetentionInDays  = config.getint     ('Debug', 'LogfileRetentionInDays')


datpart =  datetime.now().strftime('%Y-%m-%d')
# logname = str(fname) + ".log"
logname = "aiziffer." + datpart +".log"
# print (logname)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rhandler = logging.handlers.TimedRotatingFileHandler(logname,'midnight',LogfileRetentionInDays)
rhandler.namer = my_namer
rhandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s, %(name)s %(levelname)s %(message)s')
rhandler.setFormatter(formatter)

logger.addHandler(rhandler)

# logging.info("Running Urban Planning")
# logging.warning('warn message')
# logging.error('error message')
# logging.critical('critical message')

# logging.shutdown()

s = sched.scheduler (time.time, time.sleep)

# default.MaxRateValue = 1.0
# default.MaxRateType = RateChange
# default.ExtendedResolution = false
# default.IgnoreLeadingNaN = false

AutoStart         = config.getboolean ('AutoTimer', 'AutoStart')
Intervall         = config.getint     ('AutoTimer', 'Intervall')

InitialRotate      = config.getint ('Alignment', 'InitialRotate')
InitialMirror      = config.getboolean ('Alignment', 'InitialMirror')
SearchFieldX       = config.getint ('Alignment', 'SearchFieldX')
SearchFieldY       = config.getint ('Alignment', 'SearchFieldY')
FlipImageSize      = config.getboolean ('Alignment', 'FlipImageSize')


DecimalShift       = config.getint ('PostProcessing', 'main.DecimalShift')
PreValueAgeStartup = config.getint ('PostProcessing', 'PreValueAgeStartup')
PreValueUse        = config.getboolean ('PostProcessing', 'PreValueUse')
AllowNegativeRates = config.getboolean ('PostProcessing', 'AllowNegativeRates')
ErrorMessage       = config.getboolean ('PostProcessing', 'ErrorMessage')
CheckDigitIncreaseConsistency =  config.getboolean ('PostProcessing', 'CheckDigitIncreaseConsistency')

Uri                = config.get ('MQTT', 'Uri')
MainTopic          = config.get ('MQTT', 'MainTopic')
ClientID           = config.get ('MQTT', 'ClientID')
MQTTuser           = config.get ('MQTT', 'user')
MQTTpassword       = config.get ('MQTT', 'password')

Model_Dig          = config.get ('Digits', 'Model')
ModelInputSize_Dig = config.get ('Digits', 'ModelInputSize')

Model_Ana          = config.get ('Analog', 'Model')
ModelInputSize_Ana = config.get ('Analog', 'ModelInputSize')

MQTT_SERVER = Uri
MQTT_PATH   = MainTopic

img_rgb = None
img_cop = None
takt = None

anal = []
ana  = []

dig  = []
digi = []

ref_pos = []
ref_dx_dy = []
img_ref = []


mx, my = (int(j) for j in ModelInputSize_Dig.split())
# print ("Digit inpsize mx=", mx, " my=", my)
dim_dig = (mx, my)

mx, my = (int(j) for j in ModelInputSize_Ana.split())
# print ("Analog inpsize mx=", mx, " my=", my)
dim_ana = (mx, my)

# Init Models

interpreter_ana = tflite.Interpreter (model_path=Model_Ana)
input_details_ana = interpreter_ana.get_input_details()
output_details_ana = interpreter_ana.get_output_details()
# input details
# print(input_details_ana)
# output details
# print(output_details_ana)
interpreter_ana.allocate_tensors()

interpreter = tflite.Interpreter (model_path=Model_Dig)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# input details
# print(input_details)
# output details
# print(output_details)
interpreter.allocate_tensors()


def read_sections():
    for i in range (1, 10):
        digname = "main.dig" + str(i)
        try:
            entry = config.get ('Digits', digname)
            # print (digname, " ", entry)
            dig.append (entry)
        except:
            break
    # print (len(dig), " ", dig)


    for i in range (1, 10):
        ana_name = "main.ana" + str(i)
        try:
            entry = config.get ('Analog', ana_name)
            # print (ana_name, " ", entry)
            anal.append (entry)
        except:
            break
    # print (len(anal), " ", anal)

    for i in range (0, 2):
        refname = "ref" + str(i) + ".jpg"
        try:
           entry = config.get ('Alignment', refname)
           print (refname, " ", entry)
           ref_pos.append (entry)
           img_ref.append (cv2.imread(ClientID + '/' + refname))
        except:
           break
    # print (len(ref), " ", ref)
    # print (img_ref[0].shape)



def get_image ():
    global img_rgb,takt

    takt = datetime.now()
    print ("Akttime " , takt.strftime ('%Y-%m-%dT%H:%M:%S'))
    tstamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # picname =  ClientID + '_' + tstamp + ".jpg"
    # urllib.request.urlretrieve("http://192.168.178.149/capture_with_flashlight", picname)

    picname =  ClientID + '_' + tstamp + ".jpg"
    urllib.request.urlretrieve("http://192.168.178.118/capture_with_flashlight", picname)

    img_rgb = cv2.imread (picname)
    h , w  = img_rgb.shape[:-1]
    print ("Weite ", w, "Höhe ", h)

    logging.info("Image taken")


def adjust_image ():
    global img_rgb

    if InitialMirror == True:
       img_rgb = cv2.flip (img_rgb, 0)

    if FlipImageSize == True:
       img_rgb = cv2.flip (img_rgb, 1)

    if InitialRotate != 0:
       img_rgb = rotate_image (img_rgb, InitialRotate * -1)
       # cv2.imwrite('after_rotate.jpg' ,img_rgb)

    ref_dx_dy.clear()

    for i in range (0, len(ref_pos)):
        href, wref = img_ref[i].shape[:-1]
        res = cv2.matchTemplate (img_rgb, img_ref[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print ("min " ,min_val, " max ",max_val, " min_loc ", min_loc, " max_loc " , max_loc)
        print ("wref ", wref, " href", href)

        x, y = ref_pos[i].split()
        ref_dx_dy.append ((max_loc[0] - int (x) , max_loc[1] - int (y)))

        top_left = max_loc
        bottom_right = (top_left[0] + wref, top_left[1] + href)
        # print ("top_left ", top_left, " bottom_right ", bottom_right)
        cv2.rectangle (img_rgb,top_left, bottom_right, (0,0,255), 1)

        cv2.imwrite('mark_ref.jpg' ,img_rgb)

    print ("ref dx dy ", ref_dx_dy)
    logging.info("Image adjusted")


def analyse_digit():
    global img_rgb, img_cop
    digi.clear()

    if len(dig) > 0:
       img_cop = img_rgb.copy()

    for i in range (0, len (dig)):
        x, y, w, h = (int(j) for j in dig[i].split())
        # print ("x=",x," y=", y," w=", w, " h=", h)
        # print ("y=", y , " y+h=", y+h, " x=", x, "x+w=", x+w)

        img_part = img_rgb[y + ref_dx_dy[0][1]:y + h + ref_dx_dy[0][1], x + ref_dx_dy[0][0]:x +w + ref_dx_dy[0][0]]
        # img_part = img_rgb[y:y+h, x:x+w]

        top_left = (x + ref_dx_dy[0][0],y + ref_dx_dy[0][1])
        bottom_right = (x+w + ref_dx_dy[0][0], y+h + ref_dx_dy[0][1])
        cv2.rectangle (img_cop,top_left, bottom_right, (255,0,0), 1)

        cv2.imwrite ('dig_{}.bmp'.format(i), img_part)
        res_img = cv2.resize (img_part, dim_dig, interpolation = cv2.INTER_NEAREST)
        res_img = np.array (res_img, dtype="float32")
        img = np.reshape (res_img,[1,dim_dig[1], dim_dig[0],3])

        interpreter.set_tensor (input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor (output_details[0]['index'])

        if np.argmax(output_data) == 10:
           digi.append ('N')
        else:
           digi.append (np.argmax(output_data))

        cv2.imwrite ('res_{}.bmp'.format(i), res_img)

    print ("Digits :", digi)
    cv2.imwrite('mark_dig.jpg' ,img_cop)

    logging.info("Digits analysed")


def analyse_analog():
    global img_rgb, img_cop
    ana.clear()

    for i in range (0, len (anal)):
        x, y, w, h = (int(j) for j in anal[i].split())

        # print ("x=",x," y=", y," w=", w, " h=", h)
        # print ("y=", y , " y+h=", y+h, " x=", x, "x+w=", x+w)

        # img_part = img_rgb[y:y+h, x:x+w]
        img_part = img_rgb[y + ref_dx_dy[0][1]:y + h + ref_dx_dy[0][1], x + ref_dx_dy[0][0]:x + w + ref_dx_dy[0][0]]

        top_left = (x + ref_dx_dy[0][0],y + ref_dx_dy[0][1])
        bottom_right = (x + w + ref_dx_dy[0][0], y + h + ref_dx_dy[0][1])
        cv2.rectangle (img_cop,top_left, bottom_right, (0,255,0), 1)

        cv2.imwrite ('ana_{}.bmp'.format(i), img_part)
        res_img = cv2.resize (img_part, dim_ana, interpolation = cv2.INTER_NEAREST)
        res_img = np.array (res_img, dtype="float32")
        img = np.reshape (res_img,[1,dim_ana[1], dim_ana[0],3])

        interpreter_ana.set_tensor (input_details_ana[0]['index'], img)
        interpreter_ana.invoke()
        output_data = interpreter_ana.get_tensor (output_details[0]['index'])

        # print ("out ana ", output_data)
        f1 = output_data[0][0]
        f2 = output_data[0][1]
        fres = round (math.fmod (math.atan2(f1, f2) / (math.pi * 2) + 2, 1) * 10.0, 1)
        ana.append (fres)

        cv2.imwrite ('resa_{}.bmp'.format(i), res_img)

    print ("Analog :", ana)
    cv2.imwrite('mark_ana.jpg' ,img_cop)
    logging.info("Analog analysed")

def send_mqtt():
    try:
         mqtt_auth = None

         if len(MQTTuser) > 0:
            mqtt_auth = { 'username': MQTTuser, 'password': MQTTpassword }

         # auth=mqtt_auth
         publish.single (MQTT_PATH + "/main/timestamp" , takt.strftime ('%Y-%m-%dT%H:%M:%S') , hostname=MQTT_SERVER, client_id=ClientID)

         strval_dig =  ''

         for i in range (0, len (digi)):
            strval_dig = strval_dig + str(digi[i])

         if DecimalShift < 0:
            strpos = len (digi) + DecimalShift
            strval_dig = strval_dig[:strpos] + '.' + strval_dig[strpos:]
         elif DecimalShift > 0:
            strval_dig = strval_dig[:DecimalShift] + '.' + strval_dig[DecimalShift:]

         publish.single (MQTT_PATH + "/main/digit" , strval_dig , hostname=MQTT_SERVER, client_id=ClientID)


         if len(ana) > 0:
            strval_ana =  ''
            for i in range (0, len (ana)):
                strval_ana = strval_ana + str(ana[i])

            publish.single (MQTT_PATH + "/main/analog" , strval_ana , hostname=MQTT_SERVER, client_id=ClientID)

         if len(ana) > 0:
            logging.info("MQTT info sent " + strval_dig + " " + strval_ana)
         else:
            logging.info("MQTT info sent " + strval_dig)
    except:
       logging.error("MQTT Unexpected error:", sys.exc_info()[0])
       pass


def process_image(sc):
    get_image()
    adjust_image()
    analyse_digit()
    analyse_analog()
    send_mqtt()

    sc.enter(Intervall, 1, process_image, (sc,))

def main():
    read_sections()

    if AutoStart == True:
       logging.info("Timer started")
       s.enter(Intervall, 1, process_image, (s,))
       s.run()
    else:
       process_image()


if __name__ == "__main__":
    main()
    logging.shutdown()


