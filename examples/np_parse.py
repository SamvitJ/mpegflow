####
# gets a [n_frames x grid_height x grid_width x 2] motion vector numpy matrix from a video and saves to disk
# INSTRUCTIONS:
# run in mpegflow directory or add mpegflow to PATH and remove './' in os.system call
# do 'python np_parse.py --in <mpeg video> --out <pickled tensor location>'
###

import numpy as np
import argparse
import sys, os
import pickle

parser = argparse.ArgumentParser(description='Save motion vectors to numpy tensor.')
parser.add_argument('--input', required=True, type=str,
                    help='video file to get data from')
parser.add_argument('--output', required=True, type=str,
                    help='output for numpy array')
args = parser.parse_args()

motionvec_fname = '{0}.processed.txt'.format(args.input)
print('Saved to: ' + motionvec_fname)
os.system('./mpegflow {0} > {1}'.format(args.input, motionvec_fname))

motionvec_file = open(motionvec_fname, 'r')

# load all lines in motion vector file
lines = [line.strip() for line in motionvec_file.readlines()]

# parse first line for size of motion vector data per frame
first = lines[0]
shape_str = list(filter(lambda s: s.startswith('shape'), first.split(' ')))[0]
n_rows, n_cols  = shape_str[6:].split('x')
n_rows, n_cols = int(n_rows), int(n_cols)

data = []
frame_data = []
prev_frame = -1
for i in range(len(lines)):
    frame_ind = i // (n_rows + 1)
    if frame_ind != prev_frame and prev_frame != -1:
        data.append(np.array(frame_data))
        prev_frame = frame_ind
        frame_data = []
    row_ind = i % (n_rows + 1)
    # check if we are at a comment line again
    if row_ind == 0:
        continue
    else:
        # adjust for zero-indexing in matrix
        row_ind -= 1
    line = lines[i]
    row_data = map(float, line.strip().split('\t'))
    row = np.array(list(row_data))
    frame_data.append(row)
    prev_frame = frame_ind

np_data = np.array(data)

# fix to store as tensor with dx, dy channels
n_frames, n_data_rows, n_data_cols = np_data.shape
correct = np.zeros((n_frames, n_data_rows // 2, n_data_cols, 2))
correct[:, :, :, 0] = np_data[:, :n_data_rows // 2, :]
correct[:, :, :, 1] = np_data[:, n_data_rows // 2:, :]
np_data = correct

motionvec_file.close()
print('Saving [{0} x {1} x {2} x {3}] motion vector data to: \t{4}'
        .format(np_data.shape[0], np_data.shape[1], np_data.shape[2],
            np_data.shape[3], args.output))

pickle.dump(np_data, open(args.output, 'wb'))