import math
import os
import json
import cv2
import numpy as np

from ..swimmer import Swimmer
from .label import Label

from constants import Constants
from helper import Helper
from space_transform.space_transform import SpaceTransform

class SwimmingLabel(Label):
  '''
  An object representing the json labels for a swimming video. An instance of
  SwimmingLabel offers an interface for querying/constructing network targets,
  among other utility operations around transformations from image space to
  world space and evaluation of predictions
  '''
  TB_STATISTICS = []

  def __init__(self, root_dir, video_filepath, label_filepath=None, n_frames=1, skip_frames=0):
    label_filepath = label_filepath if label_filepath is not None else \
                         video_filepath[:video_filepath.rindex(".")+1] + 'json'
    super().__init__(root_dir, video_filepath, label_filepath, n_frames, skip_frames)

    self.space_transform = SpaceTransform(self.video_filepath)
    self.read_frame_info()
    self._process_json_data()

  def _process_json_data(self):
    '''
    Reads metadata from the annotation file
    - Reads all annotated swimmer coordinates and sets self.swimmers
    - Finds the first and last annotated frames, and sets self.first_frame and
      self.last_frame accordingly. If the last annotated frame is not a stop-
      frame, the video ends there; otherwise we will train on the video to the
      end. (The last frame not being a stop-frame indicates that the swimmers
      reached the end of the pool and are no longer being tracked).
    '''
    first_frame = float('inf')

    last_frame_stopframe = (-1, False)
    self.swimmers = []

    with open(self.label_filepath) as json_file:
      json_data = json.load(json_file)

    metadata = json_data[0]
    self.event_type = metadata['eventType']
    json_data = json_data[1:]
    if len(json_data) < 10:
      print("Insufficient swimmers ({}) for {}".format(len(json_data),
                                                       self.label_filepath))
    for swimmer in json_data:
      lane_number = []
      key_frames = []
      for frame_num in swimmer['keyFrames']:
        frame = int(frame_num) - 1
        position = swimmer['keyFrames'][frame_num]['pos']
        pos_x = position['x']
        pos_y = position['y']
        stop_frame = 'stopFrame' in swimmer['keyFrames'][frame_num]
        key_frames.append({
            'frame': frame,
            'x': pos_x,
            'y': pos_y,
            'stop_frame': stop_frame
        })
        if frame < first_frame:
          first_frame = frame
        if frame > last_frame_stopframe[0]:
          last_frame_stopframe = (frame, stop_frame)
      key_frames.sort(key=lambda x: x['frame'])
      self.swimmers.append(Swimmer(lane_number, key_frames))

    self.first_frame = first_frame
    if not last_frame_stopframe[1]:
      self.last_frame = last_frame_stopframe[0]
    self.frame_count = self.last_frame - self.first_frame + 1

  def screen_coords_to_world_coords(self, pred_coords):
    '''
    Given the screen-space `pred_coords`, convert them to world-space coords
    '''
    # As we downsample the video, we need to upsample our extracted coordinates
    pred_coords = [[coord[1]*Constants.config['downsample_factor'], coord[0]*Constants.config['downsample_factor']]
                   for coord in pred_coords if coord is not None]
    # Transform to world-space
    world_coords = self.space_transform.query_points(pred_coords)
    # Filter duplicate lane predictions
    return Helper.filter_duplicate_coords(world_coords)

  @staticmethod
  def _downsample_scale(the_list):
    '''
    Scales a list of numbers as according to Constants.config['downsample_factor']
    '''
    return None if the_list is None else [x / Constants.config['downsample_factor'] for x in the_list]

  def get_coords(self, frame_num):
    '''
    Returns the swimmer coordinates for a given frame number - scaled to account
    for downsampling of the video
    '''
    coords = []
    for swimmer in self.swimmers:
      coords.append(SwimmingLabel._downsample_scale(swimmer.get_interpolated_coords(frame_num)))
    return coords

  def is_valid_event_type(self, event_type):
    '''
    Returns true iff there is no restriction on event type, or this video is
    for the correct event
    '''
    if self.event_type is None:
      self._load_metadata()
    return event_type.lower() == 'all' or self.event_type.lower() == event_type.lower()

  def get_target_world_coords(self, frame_num):
    '''
    Returns the world-space coordinates for a given frame number
    '''
    coords = []
    for swimmer in self.swimmers:
      coords.append(swimmer.get_interpolated_coords(frame_num))
    result = []
    for coord in coords:
      if coord is None:
        result.append(None)
      else:
        lane, distance = self.space_transform.query_point(coord[0],
                                                          coord[1])
        if lane != -1:
          result.append((lane, distance))
        else:
          result.append(None)
    return result

  @staticmethod
  def get_blob_sprite():
    '''
    Gets (and if necessary, creates) a circular, gradient blob sprite
    '''
    blob_radius = 10
    blob_diam = blob_radius * 2
    if SwimmingLabel.blob_sprite is None:
      blob_sprite = np.zeros((blob_diam,
                              blob_diam))
      for x in range(blob_diam):
        for y in range(blob_diam):
          dist = math.sqrt(math.pow(x - blob_radius, 2) +
                           math.pow(y - blob_radius, 2)) / blob_radius
          val = int(min(255, (1. - dist)* 255))
          blob_sprite[x, y] = max(val, blob_sprite[x, y])
      SwimmingLabel.blob_sprite = blob_sprite
    return SwimmingLabel.blob_sprite, blob_radius, blob_diam

  @staticmethod
  def draw_blob(im, coords):
    '''
    Draws a blob on `im` at the given `coords`
    '''
    blob_sprite, blob_radius, blob_diam = SwimmingLabel.get_blob_sprite()
    coord_x = int(round(coords[0]))
    coord_y = int(round(coords[1]))
    for x in range(blob_diam):
      target_x = coord_x - blob_radius + x
      if target_x < 0 or target_x >= Constants.VIDEO_WIDTH:
        continue
      for y in range(blob_diam):
        target_y = coord_y - blob_radius + y
        if target_y < 0 or target_y >= Constants.VIDEO_HEIGHT:
          continue
        im[target_y, target_x] = blob_sprite[y, x]

  def draw_blobs(self, im, frame_num):
    '''
    Draws blobs on `im` at all of the coords found at frame_num.
    Modifies im in-place
    '''
    coords_list = self.get_coords(frame_num)
    for coords in coords_list:
      if coords is None:
        continue
      SwimmingLabel.draw_blob(im, coords)

  def targets(self, frame_num):
    frame_num = self._centre_frame(frame_num)
    coords_list = self.get_coords(frame_num)
    ret_coords = []
    for coords in coords_list:
      if coords is None:
        continue
      # Swap first two terms, as it should be (row, col), not (x, y)
      coords[0], coords[1] = coords[1], coords[0]
      ret_coords.append(coords)

    return ret_coords

  def evaluate(self, predictions, targets, frame_num): # pylint: disable=W0221
    '''
    Use the predictions to create list of world-space coords and compare to
    the target world-space coords.
    Returns the recall and mean_distance for the given predictions in a dict
    '''
    grid, local_coords, batch_crop_counts, global_coords, target_grid = predictions
    frame_num = self._centre_frame(frame_num[0])

    # Extract predictions which correspond to the first example
    coords = []
    if len(batch_crop_counts) > 0:
      first_batch_index, first_batch_count = batch_crop_counts[0]
      if first_batch_index == 0 and first_batch_count > 0:
        coords = global_coords[:first_batch_count]

    # Exract targets which correspond to the first example
    target_coords = []
    indices = np.reshape(np.where(targets[:, 0] == 0), -1)
    if len(indices) > 0:
      target_coords = targets[indices, 1:]

    # Transform targets and predictions to world space
    world_coords = self.screen_coords_to_world_coords(coords)
    target_world_coords = self.screen_coords_to_world_coords(target_coords)

    # Print the target/predicted lanes
    Helper.print_lanes(world_coords, target_world_coords)

    # Compute some of our metrics
    recall = Helper.compute_recall(world_coords, target_world_coords)
    mean_distance = Helper.compute_mean_world_distance(world_coords, target_world_coords)
    median_distance = Helper.compute_median_world_distance(world_coords, target_world_coords)

    print("Recall: {:.2f}%".format(recall*100) if recall is not None else "")
    print("Mean distance: {:.3f}m".format(mean_distance) if mean_distance is not None else "")
    print("Median distance: {:.3f}m".format(median_distance) if median_distance is not None else "")

    return {
        'recall': recall,
        'mean_distance': mean_distance,
        'median_distance': median_distance
    }

  def _visualise_grid(self, target_grid, pred_grid, folder, write_index):
    '''
    Saves target crop grid and the predicted crop grid for the first batch
    '''
    cv2.imwrite(os.path.join(folder, '{:02d}_pred_grid.png'.format(write_index)), pred_grid*255)
    cv2.imwrite(os.path.join(folder, '{:02d}_target_grid.png'.format(write_index)), target_grid*255)

  def _visualise_image(self, batch_crop_counts, frame, predictions, folder, write_index):
    '''
    Saves the central frame for the first batch with the predicted swimmers circled
    '''
    if len(batch_crop_counts) > 0 and batch_crop_counts[0, 0] == 0:
      for i in range(batch_crop_counts[0, 1]):
        prediction = predictions[i]
        if prediction is None:
          continue
        frame = cv2.circle(frame, (int(prediction[1]), int(prediction[0])), 30, (255, 0, 255), 3)
    cv2.imwrite(os.path.join(folder, '{:02d}_img.png'.format(write_index)), frame)

  def visualise(self, outputs, batch, folder, write_index): # pylint: disable=W0221
    '''
    Saves to a rotating buffer of image files to visualise the results
    '''
    pred_grid = outputs[0][0].squeeze()
    global_regressed = outputs[3]
    batch_crop_counts = outputs[2]

    target_grid = outputs[4][0]

    self._visualise_grid(target_grid, pred_grid, folder, write_index)
    centre_frame_num = Constants.config['n_frames']//2

    frame = np.copy(batch.frames[0][centre_frame_num])
    self._visualise_image(batch_crop_counts, frame, global_regressed, folder, write_index)
