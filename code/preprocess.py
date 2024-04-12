import numpy as np
from utils import *
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
from configs import *

dataset = 21

def imu_low_pass(imu_omega, imu_stamps):
    '''pass imu data through low pass filter'''
    fs = len(imu_stamps)/(imu_stamps[-1] - imu_stamps[0])
    sos = butter(5,LOW_PASS_BW, btype='low', fs=fs, analog=False, output='sos')
    return sosfilt(sos,imu_omega)

with np.load("../data/Encoders%d.npz" % dataset) as data:
    encoder_counts_raw = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps

with np.load("../data/Hokuyo%d.npz" % dataset) as data:
    lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
    lidar_range_min = data["range_min"]  # minimum range value [m]
    lidar_range_max = data["range_max"]  # maximum range value [m]
    lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("../data/Imu%d.npz" % dataset) as data:
    imu_omega = data["angular_velocity"][2,:]  # pick only yaw rate; angular velocity in rad/sec
    # imu_lin_acc = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

n_enc = encoder_stamps.shape[0]
n_imu = imu_stamps.shape[0]
n_lidar = lidar_stamps.shape[0]

# reshape encoder data; order is FR,FL,RR,RL
# encoder_counts = encoder_counts_raw.reshape((n_enc,4))
# calculate liner velocity from encoder data
dist_left = 0.0022/2 * (encoder_counts_raw[1] + encoder_counts_raw[3])
dist_right = 0.0022/2 * (encoder_counts_raw[0] + encoder_counts_raw[2])
dist_lin = (dist_left + dist_right)/2 # linear displacement in m
arr_tau = encoder_stamps[1:] - encoder_stamps[:-1]
v_lin = dist_lin[:-1]/arr_tau

# match lidar to encoder timestamp
list_arg_encoder_lidar_ts = []
# get closest timestamp from imu data to align to encoder ts
for ts in encoder_stamps:
    ts_enc_diff_lidar = lidar_stamps - ts
    arg_closest_ts_lidar = np.argmin(np.abs(ts_enc_diff_lidar))
    list_arg_encoder_lidar_ts.append(arg_closest_ts_lidar)
reduced_lidar_ranges = lidar_ranges[:,list_arg_encoder_lidar_ts]

# convert lidar raw values to (x,y) coordinates
# range measurement is the r value in spherical coordinates
x_mask = np.cos(np.linspace(lidar_angle_min,lidar_angle_max,lidar_ranges.shape[0]))
y_mask = np.sin(np.linspace(lidar_angle_min,lidar_angle_max,lidar_ranges.shape[0]))
x_mask_full = np.tile(x_mask,[reduced_lidar_ranges.shape[1],1]).T
y_mask_full = np.tile(y_mask,[reduced_lidar_ranges.shape[1],1]).T
# remove points too close or too far from lidar
# lidar_ranges = np.where(((lidar_ranges < LIDAR_MAX) & (lidar_ranges > LIDAR_MIN)),lidar_ranges,np.nan)
x_lidar = reduced_lidar_ranges * x_mask_full
y_lidar = reduced_lidar_ranges * y_mask_full

# transform lidar frame to body frame; add this matrix to body to world pose matrices from kinematics model
bTl = np.zeros((3,3))
bTl[0:2,0:2] = np.eye(2)
bTl[0,2] = LIDAR_X_OFFSET
bTl[2,2] = 1

# match imu and encoder time stamps; needed for kinematic model
list_arg_encoder_ts = []
# get closest in past timestamp from imu data to align to encoder ts
for ts in encoder_stamps:
    ts_imu_diff = imu_stamps - ts
    arg_closest_past_ts = len(ts_imu_diff[ts_imu_diff <= 0]) - 1
    # if arg_closest_past_ts < n_enc:
    list_arg_encoder_ts.append(arg_closest_past_ts)
# apply low pass filter to imu data
imu_filt = imu_low_pass(imu_omega, imu_stamps)
reduced_imu = imu_filt[list_arg_encoder_ts]


