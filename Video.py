import os
import glob
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

import torch
import dlib
import cv2 as cv


# http://spandh.dcs.shef.ac.uk/gridcorpus/ - GRID dataset
# https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ - IBUG dataset
# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/?_ga=2.191050858.1913902486.1642077624-847301844.1639919914 - face recognition tutorial
# https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2 - facial landmarks models
# https://github.com/rizkiarm/LipNet - LipNet - Github

'''
dlib face detector and shape_predictor_68_face_landmarks landmarks predictor.
'''
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(cfg.PRETRAINED_SHAPE_DETECTOR_PATH)

class VideoReader:
    '''
    Reads video frames from .mpg file.
    '''
    def __init__(self, path):
        '''
        Extract frames from .mpg file.
        '''
        cap = cv.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # convert the image to grey scale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        self.frames = np.array(frames)
        self.video_tensor = torch.Tensor(self.frames)

    def get_tensor(self):
        '''
        Returns frames as pytorch tensor.
        '''
        self.video_tensor = self.video_tensor.unsqueeze(dim=1)
        return self.video_tensor

    def get_frames(self):
        '''
        Returns frames as numpy array.
        '''
        return self.frames


class Padder:
    '''
    Padds landmarks to find bounding box of the lips.
    '''
    def __init__(self, method, h_pad, v_pad):
        '''
        Args:
        method: str, no-pad/precentage-padding/pixel-padding/fixed-size-padding
        h_pad: int/float, horizontal pad.
        v_pad: int/float, vertical pad.
        '''
        self.method = method
        self.h_pad = h_pad
        self.v_pad = v_pad

    def pad(self, np_points):
        '''
        no-pad: no pad is using.
        precentage-padding: padding with precentage of min/max.
        pixel-padding: padding the min/max points with fixed number of pixels.
        fixed-size-padding: padding the centriod with fixed size bounding box - recomended.
        '''
        xs=np_points[:, :-1]
        ys=np_points[:, 1:]

        centroid = np.mean(np_points, axis=0)

        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)
        
        if self.method=='no-pad':
            left = min_x
            right = max_x
            top = min_y
            bottom = max_y

        if self.method=='precentage-padding':
            left = min_x * (1.0 - self.h_pad)
            right = max_x * (1.0 + self.h_pad)
            top = min_y * (1.0 - self.v_pad)
            bottom = max_y * (1.0 + self.v_pad)
            
        if self.method=='pixel-padding':
            left = min_x - self.h_pad
            right = max_x + self.h_pad
            top = min_y - self.v_pad
            bottom = max_y + self.v_pad

        if self.method=='fixed-size-padding':
            left = centroid[0] - self.h_pad / 2
            right = centroid[0] + self.h_pad / 2
            top = centroid[1] - self.v_pad / 2
            bottom = centroid[1] + self.v_pad / 2

        return int(left), int(right), int(top), int(bottom)

class Rectangle:
    '''
    Indicated bounding box.
    '''
    def __init__(self, rect):
        '''
        Args:
        rect: dlib.Rectangle, output of face detector.
        '''
        self.rect = rect
        bl = rect.bl_corner()
        tr = rect.tr_corner()
        self.bl = (bl.x, bl.y)
        self.tr = (tr.x, tr.y)

    def get_points(self):
        '''
        Returns bounding box coordinates.
        '''
        return np.array([self.bl[0], self.bl[1], self.tr[0], self.tr[1]])

    def copy(self):
        '''
        Returns copy of self.
        '''
        return Rectangle(self.rect)
    
    def is_close_to(self, other_rect, num_pixles):
        '''
        Checks if 2 Rectangles coordicnates are num_pixels close.
        Args:
        other_rect: Rectangle, compared rectangle.
        num_pixles: int, maximum distance between rectangles, number of pixels.
        '''
        points1 = self.get_points()
        points2 = other_rect.get_points()
        diff = points1 - points2
        diff = np.abs(diff)
        diff = diff > num_pixles
        far_rects_flag = not diff.any()
        return far_rects_flag

class Video:
    '''
    Process videos:
    - loads frames.
    - finds mouth bounding boxes.
    - finds mouth landmarks.
    - plots bounded lips.
    - plots scattered landmarks.
    '''
    def __init__(self, video_path):
        '''
        Ars:
        video_path: str, path to video.
        '''
        self.face_predictor_path = cfg.PRETRAINED_SHAPE_DETECTOR_PATH
        self.detector = DETECTOR
        self.predictor = PREDICTOR

        self.video_path = video_path
        self.frames = self._frames_from_video()
        

    def _frames_from_video(self):
        '''
        Process frames using VideoReader.
        '''
        reader = VideoReader(path=self.video_path)
        frames = reader.get_frames()
        return frames

    def super_fast_find_mouth_frames(self):
        '''
        Finds mouth frames using only one face bounding box,
        and landmark predictions for each frame.
        Results will be saves on self.mouth_frames.
        '''
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 70

        # middle frame face detection:
        rects=[]
        middle_frame_idx = int(len(self.frames)/2)
        mf_indices = (np.arange(0,len(self.frames), 1) + middle_frame_idx)%len(self.frames)

        # detect middle frame face. if can't, search in middle+1 frame, ...
        for i in mf_indices:
            face_detected_frame = self.frames[i]
            rects = self.detector(face_detected_frame,1)
            if len(rects) == 0:
                continue
            shape = None
            shape = self.predictor(face_detected_frame, rects[0]) # shape.parts() is 68 (x,y) points
            if shape:
                break
        
        # if no landmarks found
        if shape is None:
            print('Warning: No Landmarks Found.')

        # padding options.
        # currently the fixed-size-padding padding is used.
        padder = Padder(method='no-pad', h_pad=0, v_pad=0)
        padder = Padder(method='precentage-padding', h_pad=0.15, v_pad=0.15)
        padder = Padder(method='pixel-padding', h_pad=10, v_pad=10)
        padder = Padder(method='fixed-size-padding', h_pad=MOUTH_WIDTH, v_pad=MOUTH_HEIGHT) # creates lips with the form: width=h_pad, height=v_pad

        mouth_points = [(part.x, part.y) for part in shape.parts()[48:]] # points 48-64 indicate the mouth region
        np_mouth_points = np.array(mouth_points)

        mouth_left, mouth_right, mouth_top, mouth_bottom = padder.pad(np_mouth_points)

        mouth_frames = []
        for frame in self.frames:
            mouth_crop_image = frame[mouth_top:mouth_bottom, mouth_left:mouth_right]
            mouth_frames.append(mouth_crop_image)
        self.mouth_frames = np.array(mouth_frames)
        

    def find_mouth_frames(self, verbose=False, efficient=True):
        '''
        Find mouth frames, see 'efficient' argument.
        Results will be saves on self.mouth_frames.
        Args:
        verbose: bool, print bounding box.
        efficient: bool, if 2 face bounding box are close,
                   take the same lips-bounding-box coordinates
                   as last frame.
        '''
        MOUTH_WIDTH = 80
        MOUTH_HEIGHT = 50

        # padding options.
        # currently the fixed-size-padding padding is used.
        padder = Padder(method='no-pad', h_pad=0, v_pad=0)
        padder = Padder(method='precentage-padding', h_pad=0.15, v_pad=0.15)
        padder = Padder(method='pixel-padding', h_pad=10, v_pad=10)
        padder = Padder(method='fixed-size-padding', h_pad=MOUTH_WIDTH, v_pad=MOUTH_HEIGHT) # creates lips with the form: width=h_pad, height=v_pad

        mouth_frames = []
        last_face = None
        last_crop_points = None
        for frame in self.frames:
            mouth_tup = self._find_mouth(frame, padder, last_face, last_crop_points, verbose, efficient)
            mouth_crop_image, last_face, last_crop_points = mouth_tup
            mouth_frames.append(mouth_crop_image)
        
        self.mouth_frames = np.array(mouth_frames)

    def _find_mouth(self, frame, padder, last_face, last_crop_points, verbose=False, efficient=True):
        '''
        Find mouth bounding box in frame.
        Args:
        frame: np.array, a frame of a person.
        padder: Padder.
        last_face: Rectangle, last face bounding box.
        last_crop_points: tuple, last lips bounding box coordinates.
        verbose: bool, if True, print the lips bounding box.
        efficient: bool, if the last face bounding box is close
                   to the current fame bounding box, take the same
                   lips-bounding-box coordinates as last frame.
        '''
        # Face detection:
        rects = self.detector(frame,1)
        if len(rects) == 0:
            return frame
        
        # Check if last face is close to the current face
        current_face = Rectangle(rects[0])
        if efficient and last_face and current_face.is_close_to(last_face, 5):
            mouth_left, mouth_right, mouth_top, mouth_bottom = last_crop_points
            mouth_crop_image = frame[mouth_top:mouth_bottom, mouth_left:mouth_right]
            return mouth_crop_image, last_face, last_crop_points
        else:
            # The last face isn't close to the current face, so we need to recompute
            # the landmarks and find the lips with new 68 landmarks.
            shape = None
            shape = self.predictor(frame, rects[0]) # shape.parts() is 68 (x,y) points
            if shape is None:  # Detector doesn't detect face, return the original frame
                return frame

            mouth_points = [(part.x, part.y) for part in shape.parts()[48:]] # points 48-64 indicate the mouth region
            np_mouth_points = np.array(mouth_points)

            mouth_left, mouth_right, mouth_top, mouth_bottom = padder.pad(np_mouth_points)
            current_crop_points = mouth_left, mouth_right, mouth_top, mouth_bottom
            mouth_crop_image = frame[mouth_top:mouth_bottom, mouth_left:mouth_right]
            if verbose:
                print(mouth_left, mouth_right, mouth_top, mouth_bottom)
            return mouth_crop_image, current_face, current_crop_points


    def find_landmarks(self):
        '''
        Find landmarks for each frame using only the middle frame's face detection.
        '''
        # middle frame face detection:
        rects=[]
        middle_frame_idx = int(len(self.frames)/2)
        mf_indices = (np.arange(0,len(self.frames), 1) + middle_frame_idx)%len(self.frames)

        # detect middle frame face. if can't, search in middle+1 frame, ...
        for i in mf_indices:
            face_detected_frame = self.frames[i]
            rects = self.detector(face_detected_frame,1)
            if len(rects) > 0:
                break
        
        # if no face detected.
        if len(rects) == 0:
            print('########################################')
            print('NO FACE DETECTED!')
            print(f'VIDEO PATH: {self.video_path}')
            print('########################################')
            self.mouth_frames = None
            self.mouth_lmrks = None
        
        else:
            mouth_lmrks = []
            for i, frame in enumerate(self.frames):
                # compute 68 landmark.
                shape = None
                shape = self.predictor(frame, rects[0]) # shape.parts() is 68 (x,y) points
                if shape is None:  # Predictor can't find landmarks.
                    print(f"FRAME {i}: Predictor can't find face landmarks.")
                    mouth_lmrks.append(np.array([(0,0)]*20))

                mouth_points = [(part.x, part.y) for part in shape.parts()[48:]] # points 48-68 indicate the mouth region
                # mouth_points = [(part.x, part.y) for part in shape.parts()[:]] # return all points

                np_mouth_points = np.array(mouth_points)
                mouth_lmrks.append(np_mouth_points)
                
            self.mouth_lmrks = np.array(mouth_lmrks)


    def plot_random_lips(self, fig_path=None):
        '''
        Plots original frame and its lips bounding box.
        Args:
        fig_path: str, path to save figure.
        '''
        frame_index = random.randrange(len(self.frames))
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(self.frames[frame_index], cmap='gray')
        axs[0].set_title('Original Frame')
        axs[1].imshow(self.mouth_frames[frame_index], cmap='gray')
        axs[1].set_title('Lips Crop')
        if fig_path:
            plt.savefig(fig_path)

    def plot_random_lips_and_lmrks(self, fig_path=None):
        '''
        Plots random original frame, its landmarks scattered,
        and its lips bounding box.
        Args:
        fig_path: str, path to save figure.
        '''
        frame_index = random.randrange(len(self.frames))
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        axs[0].imshow(self.frames[frame_index], cmap='gray')
        axs[0].set_title('Original Frame')
        axs[1].imshow(self.frames[frame_index], cmap='gray')
        axs[1].scatter(self.mouth_lmrks[frame_index][:,:-1], self.mouth_lmrks[frame_index][:,1:], s=1)
        axs[1].set_title('Lips Landmarks')
        axs[2].imshow(self.mouth_frames[frame_index], cmap='gray')
        axs[2].set_title('Lips Bounding Box')
        for ax in axs:
            ax.axis('off')
        if fig_path:
            plt.savefig(fig_path)

    def __repr__(self):
        try:
            mouth_frames_shape = np.array(self.mouth_frames)[0].shape
        except:
            mouth_frames_shape = 'unequal'
        return f'frames shape: {self.frames.shape}\n' + f'mouth frames: {len(self.mouth_frames)}, frame shape: {mouth_frames_shape}'

class LandmarksCompressor:
    '''
    Preprocesses videos and compresses their lips landmarks.
    '''
    def __init__(self, original_path, compressed_path):
        '''
        Args:
        original_path: str, videos directory.
        compressed_path: str, output directory. 
        '''
        self.original_path = original_path
        self.compressed_path = compressed_path
        self.video_paths = glob.glob(self.original_path + '/*/*')

        for vid_path in tqdm(self.video_paths):
            vid = Video(vid_path)
            vid.find_landmarks()
            data = vid.mouth_lmrks
            if data is None:
                 continue
            new_path = '/'.join(vid_path.split('/')[-2:])
            file_path = self.compressed_path + '/' + new_path[:-4] + '.npy'
            dir_path = '/'.join(file_path.split('/')[:-1])
            os.makedirs(dir_path, exist_ok=True)
            file = open(file_path, 'w+') # create file
            file.close()
            file = open(file_path, 'wb')
            np.save(file, data, allow_pickle=True)
            file.close()