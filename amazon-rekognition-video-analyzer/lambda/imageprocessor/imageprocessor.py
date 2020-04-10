# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Amazon Software License (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#     http://aws.amazon.com/asl/
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and limitations under the License.

from __future__ import print_function
import base64
import datetime
import time
import decimal
import uuid
import json
import pickle
import boto3
import pytz
from pytz import timezone
from copy import deepcopy
import cv2
import numpy as np


def load_config():
    '''Load configuration from file.'''
    with open('imageprocessor-params.json', 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

def replace_float(obj):
    if isinstance(obj, list):
        for i in xrange(len(obj)):
            obj[i] = replace_float(obj[i])
            return obj
    elif isinstance(obj, dict):
        for k in iter(obj.keys()):
            obj[k] = replace_float(obj[k])
            return obj
    elif isinstance(obj, float):
        if obj % 1 == 0:
            return int(obj)
        else:
            return decimal.Decimal(str(obj))
    else:
        return obj

def convert_ts(ts, config):
    '''Converts a timestamp to the configured timezone. Returns a localized datetime object.'''
    #lambda_tz = timezone('US/Pacific')
    tz = timezone(config['timezone'])
    utc = pytz.utc

    utc_dt = utc.localize(datetime.datetime.utcfromtimestamp(ts))

    localized_dt = utc_dt.astimezone(tz)

    return localized_dt


def variance_of_laplacian(image): #Added this
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_frame_blur_light(frame): #Added this
	blur_threshold=100
	light_lower = 100
	light_upper = 150
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	fm = variance_of_laplacian(frame_gray)
	blur_text = "Not Blurry"

	if fm < blur_threshold:
		blur_text = "Blurry"

	light_text = 'Good Lighting '
	mean, std = cv2.meanStdDev(frame_gray)
	mean = mean[0][0]
	std = std[0][0]
	if mean > light_upper:
		light_text = "Too Bright"
	elif mean < light_lower:
		light_text = 'Too Dark'

	print("{}: {:.2f}".format(blur_text, fm))
	print("{}: {:.2f}".format(light_text, mean))
	return blur_text, fm, light_text, mean



def process_image(event, context):

    #Initialize clients
    rekog_client = boto3.client('rekognition')
    sns_client = boto3.client('sns')
    s3_client = boto3.client('s3')
    #dynamodb = boto3.resource('dynamodb')
    dynamodb = boto3.resource('dynamodb')

    #Load config
    config = load_config()

    s3_bucket = config["s3_bucket"]
    s3_key_frames_root = config["s3_key_frames_root"]

    ddb_table = dynamodb.Table(config["ddb_table"])

    rekog_max_labels = config["rekog_max_labels"]
    rekog_min_conf = float(config["rekog_min_conf"])

    label_watch_list = config["label_watch_list"]
    label_watch_min_conf = float(config["label_watch_min_conf"])
    label_watch_phone_num = config.get("label_watch_phone_num", "")
    label_watch_sns_topic_arn = config.get("label_watch_sns_topic_arn", "")

    #Iterate on frames fetched from Kinesis
    for record in event['Records']:

        frame_package_b64 = record['kinesis']['data']
        frame_package = pickle.loads(base64.b64decode(frame_package_b64))

        img_bytes = frame_package["ImageBytes"]
        approx_capture_ts = frame_package["ApproximateCaptureTime"]
        frame_count = frame_package["FrameCount"]

        now_ts = time.time()

        frame_id = str(uuid.uuid4())
        processed_timestamp = decimal.Decimal(now_ts)
        approx_capture_timestamp = decimal.Decimal(approx_capture_ts)

        now = convert_ts(now_ts, config)
        year = now.strftime("%Y")
        mon = now.strftime("%m")
        day = now.strftime("%d")
        hour = now.strftime("%H")

        rekog_response = rekog_client.detect_labels(
            Image={
                'Bytes': img_bytes
            },
            MaxLabels=rekog_max_labels,
            MinConfidence=rekog_min_conf
        )


        #Iterate on rekognition labels. Enrich and prep them for storage in DynamoDB
        #labels_on_watch_list = []
        new_labels = []
        for label in rekog_response['Labels']:

            lbl = label['Name']
            conf = label['Confidence']
            label['OnWatchList'] = False

            #Print labels and confidence to lambda console
            print('{} .. conf %{:.2f}'.format(lbl, conf))

            #Check label watch list and trigger action
            #if (lbl.upper() in (label.upper() for label in label_watch_list)
                #and conf >= label_watch_min_conf):

                #label['OnWatchList'] = True
                #labels_on_watch_list.append(deepcopy(label))

            #Convert from float to decimal for DynamoDB
            label['Confidence'] = decimal.Decimal(str(conf))

            new_label = {}
            new_label['Name'] = lbl
            new_label['Confidence'] = decimal.Decimal(str(conf))
            new_labels.append(new_label)

        #Perform blur/light detection
        hi = np.asarray(img_bytes, dtype="uint8")
        image = cv2.imdecode(hi, cv2.IMREAD_COLOR)
        blur_text, fm, light_text, mean = compute_frame_blur_light(image)
        new_label = {}
        new_label['Name'] = str(blur_text)
        new_label['Confidence'] = decimal.Decimal(str(fm))
        new_labels.append(new_label)

        new_label = {}
        new_label['Name'] = str(light_text)
        new_label['Confidence'] = decimal.Decimal(str(mean))
        new_labels.append(new_label)

        # #Send out notification(s), if needed
        # if len(labels_on_watch_list) > 0 \
        #         and (label_watch_phone_num or label_watch_sns_topic_arn):
        #
        #     notification_txt = 'On {}...\n'.format(now.strftime('%x, %-I:%M %p %Z'))
        #
        #     for label in labels_on_watch_list:
        #
        #         notification_txt += '- "{}" was detected with {}% confidence.\n'.format(
        #             label['Name'],
        #             round(label['Confidence'], 2))
        #
        #     print(notification_txt)
        #
        #     if label_watch_phone_num:
        #         sns_client.publish(PhoneNumber=label_watch_phone_num, Message=notification_txt)
        #
        #     if label_watch_sns_topic_arn:
        #         resp = sns_client.publish(
        #             TopicArn=label_watch_sns_topic_arn,
        #             Message=json.dumps(
        #                 {
        #                     "message": notification_txt,
        #                     "labels": labels_on_watch_list
        #                 }
        #             )
        #         )
        #
        #         if resp.get("MessageId", ""):
        #             print("Successfully published alert message to SNS.")

        #Store frame image in S3
        s3_key = (s3_key_frames_root + '{}/{}/{}/{}/{}.jpg').format(year, mon, day, hour, frame_id)

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=img_bytes
        )

        #Persist frame data in dynamodb

        item = {
            'frame_id': frame_id, #string
            'processed_timestamp' : processed_timestamp, #decimal
            'approx_capture_timestamp' : approx_capture_timestamp, #decimal
            #'rekog_labels' : rekog_response['Labels'],
            'rekog_labels' : new_labels,
            #'rekog_labels' : ['cat', 'mouse'],
            #'rekog_orientation_correction' : 'ROTATE_0',
            'processed_year_month' : year + mon, #To be used as a Hash Key for DynamoDB GSI
            's3_bucket' : s3_bucket,
            's3_key' : s3_key
        }

        new_item = replace_float(item)

        ddb_table.put_item(Item=new_item)

    print('Successfully processed {} records.'.format(len(event['Records'])))
    return

def handler(event, context):
    return process_image(event, context)
