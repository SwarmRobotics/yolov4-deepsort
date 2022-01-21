#!/usr/bin/env python3

import os
import cv2
from core.E2P import E2P
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import global_variables_initializer

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageMsg
from jackalnet_ros.msg import JackalNetDetection
import message_filters

import rospkg

JACKALNET_PACKAGE_PATH = rospkg.RosPack().get_path("jackalnet_ros")

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/jackalnet-416-best',
                    'path to weights file') #'./checkpoints/yolov4-416'
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam') #'./data/video/test.mp4'
flags.DEFINE_string('output', '/home/noah/JackalNet Final/Rendered.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

class FLAGS():
    framework = 'tf'
    weights = JACKALNET_PACKAGE_PATH + '/src/checkpoints/jackalnet-416-best'
    size = 416
    tiny = False
    model = 'yolov4'
    video = '0' #'./data/video/test.mp4'
    output = '/home/noah/JackalNet Final/Rendered.mp4'
    output_format = 'XVID'
    iou = 0.45
    score = 0.50
    dont_show = False
    info = False
    count = False
    publish_image = True

def Main():
    rospy.logwarn("Initialising object detection")
    bridge = CvBridge()

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = JACKALNET_PACKAGE_PATH + '/src/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        rospy.loginfo(input_details)
        rospy.loginfo(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    # try:
    #     vid = cv2.VideoCapture(int(video_path))
    # except:
    #     vid = cv2.VideoCapture(video_path)

    out = None

    # Setup Perspectives
    perspective_instances = []
    noPerspectives = 4
    frame_w = 3840
    frame_h = 2160
    height = 600
    width = 1200
    RADIUS = 128
    ScoreReductionThreshold = 0.1
    ScoreReductionFactor = 0.8

    #AUTOMATIC PERSPECTIVE GENERATION
    for i in range(0, noPerspectives):
        perspective_instances.append(E2P(frame_w, frame_h, 114, 70, ((360 / noPerspectives) * i)-90, 0, height, width, RADIUS))

    #perspective_instances.append(E2P(frame_w, frame_h, 110, 70, 0, 0, height, width))
    #perspective_instances.append(E2P(frame_w, frame_h, 110, 70, 90, 0, height, width))
    #perspective_instances.append(E2P(frame_w, frame_h, 110, 70, 180, 0, height, width))
    #perspective_instances.append(E2P(frame_w, frame_h, 110, 70, -90, 0, height, width))


    # get video ready to save locally if flag is set

    # if FLAGS.output:
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(vid.get(cv2.CAP_PROP_FPS))
    #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # while video is running
    rospy.logwarn("Initialisation complete, processing frames")
    while True:
        # images = [rospy.wait_for_message('camera/full/image_raw', ImageMsg),
        #           rospy.wait_for_message('camera/front/image_rect', ImageMsg),
        #           rospy.wait_for_message('camera/right/image_rect', ImageMsg),
        #           rospy.wait_for_message('camera/back/image_rect', ImageMsg),
        #           rospy.wait_for_message('camera/left/image_rect', ImageMsg)]

        frame = bridge.imgmsg_to_cv2(images[0], "rgb8")
        result = []
        for perspective in images[1:]:
            result.append(bridge.imgmsg_to_cv2(perspective, "rgb8"))

        # return_value, frame = vid.read()

        # result = []
        # if return_value:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     for perspective in perspective_instances:
        #         result.append(perspective.Remap(frame))
        #     image = Image.fromarray(frame)
        # else:
        #     print('Video has ended or failed, try a different video format!')
        #     break

        #cv2.imshow('0', cv2.resize(cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR), (800, 400)))
        #cv2.imshow('1', cv2.resize(cv2.cvtColor(result[1], cv2.COLOR_RGB2BGR), (800, 400)))
        #cv2.imshow('2', cv2.resize(cv2.cvtColor(result[2], cv2.COLOR_RGB2BGR), (800, 400)))
        #cv2.imshow('3', cv2.resize(cv2.cvtColor(result[3], cv2.COLOR_RGB2BGR), (800, 400)))

        image_data = []
        for i in range(len(result)):
            frame_size = frame.shape[:2]
            image_data.append(cv2.resize(result[i], (input_size, input_size)))
            image_data[i] = image_data[i] / 255.
            image_data[i] = image_data[i][np.newaxis, ...].astype(np.float32)
            start_time = time.time()

        panoramabboxes = []
        panoramascores = []
        panoramaclasses = []
        panoramanumobjects = 0
        for count, image_data in enumerate(image_data):
            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )



            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            #Lower Scores for predictions near frame edges
            for idx, box in enumerate(bboxes):
                if (box[1] < ScoreReductionThreshold) or (box[3] > (1-ScoreReductionThreshold)):
                    scores[idx] = scores[idx] * ScoreReductionFactor

            #Reproject onto total
            if not bboxes.size == 0:
                panoramabboxes.append(perspective_instances[count].Sphere2PointArray(perspective_instances[count].Rect2SphereArray(bboxes))[0])
                panoramascores = np.append(panoramascores, scores)
                panoramaclasses = np.append(panoramaclasses, classes)
                panoramanumobjects = panoramanumobjects+num_objects

        panoramabboxes = perspective_instances[0].format_boxes(panoramabboxes)
        pred_bbox = np.empty([len(panoramabboxes), 6])
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        for idx2, detect in enumerate(panoramabboxes):
            pred_bbox[idx2] = np.array([detect[1] % frame_w, detect[0], detect[3] % frame_w, detect[2], panoramascores[idx2], panoramaclasses[idx2]])

        #Run NMS on panorama
        rospy.logdebug("Before NMS size: {}".format(len(pred_bbox)))
        pred_bbox = utils.nmsDEEPSORT(pred_bbox, 0.1, method='nms')
        rospy.logdebug("After NMS size: {}".format(len(pred_bbox)))

        #Slice out to original arrays
        panoramanumobjects = len(pred_bbox)

        panoramabboxes = np.empty([len(pred_bbox), 4])
        for idx2, detect in enumerate(pred_bbox):
            panoramabboxes[idx2] = np.array([detect[1], detect[0], detect[3], detect[2]])
        panoramascores = np.empty([len(pred_bbox)])
        for idx2, score in enumerate(pred_bbox):
            panoramascores[idx2] = score[4]
        panoramaclasses = np.empty([len(pred_bbox)])
        for idx2, Class in enumerate(pred_bbox):
            panoramaclasses[idx2] = Class[5]

        original_h, original_w, _ = frame.shape
        panoramabboxes = utils.format_boxes(panoramabboxes, original_h, original_w)



        # store all predictions in one parameter for simplicity when calling functions
        #pred_bbox = [panoramabboxes, panoramascores, panoramaclasses, panoramanumobjects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(panoramanumobjects):
            class_indx = int(panoramaclasses[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            rospy.loginfo("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        panoramabboxes = np.delete(panoramabboxes, deleted_indx, axis=0)
        panoramascores = np.delete(panoramascores, deleted_indx, axis=0)


        # encode yolo detections and feed to tracker
        features = encoder(frame, panoramabboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(panoramabboxes, panoramascores, names, features)]
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            #Calculate Relative Angles
            relativeCoords = E2P.Coord2WorldStatic([((bbox[2] + bbox[0]) / 2), ((bbox[1] + bbox[3]) / 2)], frame_w, frame_h)
            bbox_coord = '%s: %.2f %s: %.2f' % ('TH', relativeCoords[0], 'PH', relativeCoords[1])

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            #Top Description
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            #Bottom Description
            cv2.circle(frame, (int((bbox[0]+bbox[2])/2), (int((bbox[1]+bbox[3])/2))), 4, color, -1)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[3])),
                            (int(bbox[0]) + (len(bbox_coord)) * 17, int(bbox[3]+30)), color, -1)
            cv2.putText(frame, bbox_coord, (int(bbox[0]), int(bbox[3] + 20)), 0, 0.75, (255, 255, 255), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                rospy.loginfo("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # publish to ros topic
            detection_msg = JackalNetDetection()
            detection_msg.header.stamp = rospy.Time.now()
            detection_msg.track_id = track.track_id
            detection_msg.azimuth = relativeCoords[0]
            detection_msg.elevation = relativeCoords[1]
            detection_msg.xmin = int(bbox[0])
            detection_msg.ymin = int(bbox[1])
            detection_msg.xmax = int(bbox[2])
            detection_msg.ymax = int(bbox[3])
            detection_publisher.publish(detection_msg)

            

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        rospy.logdebug("FPS: %.2f" % fps)
        if FLAGS.publish_image:
            image_publisher.publish(bridge.cv2_to_imgmsg(frame, "rgb8"))
        # result = np.asarray(frame)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if not FLAGS.dont_show:
        #     cv2.imshow("Output Video", cv2.resize(result, (1920, 1080)))

        # # if output flag is set, save video file
        # if FLAGS.output:
        #     out.write(result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'): break
    # cv2.destroyAllWindows()

def callback(*args):
    global images
    images = args


if __name__ == '__main__':
    try:
        rospy.init_node("jackalnet")

        raw_sub = message_filters.Subscriber('camera/full/image_raw', ImageMsg)
        left_sub = message_filters.Subscriber('camera/left/image_rect', ImageMsg)
        front_sub = message_filters.Subscriber('camera/front/image_rect', ImageMsg)
        right_sub = message_filters.Subscriber('camera/right/image_rect', ImageMsg)
        back_sub = message_filters.Subscriber('camera/back/image_rect', ImageMsg)

        ts = message_filters.TimeSynchronizer([raw_sub, left_sub, front_sub, right_sub, back_sub], 1)
        ts.registerCallback(callback)
        detection_publisher = rospy.Publisher('jackalnet_detections', JackalNetDetection, queue_size=10)
        if FLAGS.publish_image:
            image_publisher = rospy.Publisher('jackalnet_detections_image', ImageMsg, queue_size=10)
        Main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted")
        pass
