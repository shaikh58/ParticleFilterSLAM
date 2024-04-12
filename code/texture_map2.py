import numpy as np
import matplotlib.pyplot as plt
import cv2
from configs import *

dataset=21
res = 0.05
def normalize(img):
 max_ = img.max()
 min_ = img.min()
 return (img - min_)/(max_-min_)

with np.load("../data/Kinect%d.npz"%dataset) as data:
  disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
  rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

# trajectories have been matched to the encoder timestamps so match image to encoder as well
with np.load("../data/Encoders%d.npz" % dataset) as data:
    encoder_counts_raw = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps

# load trajectory from particle filter output
arr_poses = np.load('../data/particle_filter_poses_n=20.npy')

# now match the disparity timestamps to the rgb timestamps
list_arg_encoder_rgb_ts = []
# get closest timestamp from imu data to align to encoder ts
for ts in rgb_stamps:
    ts_enc_diff_kinect = encoder_stamps - ts
    arg_closest_ts_kinect = np.argmin(np.abs(ts_enc_diff_kinect))
    list_arg_encoder_rgb_ts.append(arg_closest_ts_kinect)

reduced_pose_arr = arr_poses[list_arg_encoder_rgb_ts]

# construct disparity camera to world transformation
psi = 0.021
theta = 0.36
R_z = np.array([[np.cos(psi),-np.sin(psi),0],
                [np.sin(psi),np.cos(psi),0],
                [0,0,1]])
R_y = np.array([[np.cos(theta),0,np.sin(theta)],
                [0,1,0],
                [-np.sin(theta),0,np.cos(theta)]])
x_offset = (115.1 + 2.66)/1000
z_offset = 380.1/1000
p_cam_in_body = np.array([x_offset,0,z_offset])
bTc = np.zeros((4,4))
bTc[0:3,0:3] = R_y @ R_z # camera to body rotation matrix
bTc[3,3] = 1
bTc[0:3,3] = p_cam_in_body
oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
oRr_inv = np.linalg.inv(oRr)
K = np.array([[585.05108211,0,242.94140713],
                          [0,585.05108211,315.83800193],
                          [0,0,1]])
k_inv = np.linalg.inv(K)
z_threshold = 0.2
occ_grid_rgb = np.zeros((1130, 1130, 3))
grid_centre_x = int(1130 / 2 - 1)
grid_centre_y = int(1130 / 2 - 1)

# load images
for i in range(rgb_stamps.shape[0]):
# for i in range(1000):
    if i%100 == 0:
        print(i)
    filename = "../data/dataRGBD/RGB{y}/rgb{y}_{x}".format(y=dataset,x=i+1)
    imc = cv2.imread(filename+'.png')[...,::-1]
    filename_disp = "../data/dataRGBD/Disparity{y}/disparity{y}_{x}".format(y=dataset,x=i+1)
    imd = cv2.imread(filename_disp+'.png', cv2.IMREAD_UNCHANGED)
    disparity = imd.astype(np.float32)

    # transformation from rgb frame to disparity frame
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]

    # get 3D coordinates 
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)*z
    rgbv = np.round((v * 526.37 + 16662.0)/fy)*z
    rgb_stacked = np.stack((rgbu,rgbv,np.ones((rgbu.shape))*z),axis=2)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

    # convert robot pose to 4x4
    a = reduced_pose_arr[i,0]
    aprime = np.vstack((a.T, np.array([0, 0, 1]))).T
    conv_pose = np.vstack((aprime[:,[0, 1, 3, 2]],np.array([0,0,0,1])))
    conv_pose[2,3] = 0
        # wTc = np.matmul(conv_pose,bTc)
        #
        # top_left = np.matmul(oRr,wTc[0:3,0:3].T)
        # top_right = np.matmul(-oRr,wTc[0:3,0:3].T) @ wTc[0:3,2]
        # oTw = np.zeros((4,4))
        # oTw[0:3,0:3] = top_left
        # oTw[0:3,3] = top_right
        # oTw[3,3] = 1

    xyz_optical = np.einsum('ji,mni -> mnj',k_inv,rgb_stacked)
    # convert optical to regular camera frame
    xyz_regular = np.einsum('ji,mni -> mnj',oRr_inv,xyz_optical)
    # convert camera frame to body
    xyz_regular_hom = np.vstack((np.ones((1, 640, 480)), xyz_regular.T)).T[:,:,[1,2,3,0]]
    xyz_body = np.einsum('ji,mni -> mnj',bTc,xyz_regular_hom)
    # convert body frame to world
    xyz_world = np.einsum('ji,mni -> mnj',conv_pose,xyz_body)
    x_inds, y_inds = np.where((xyz_world[:, :, 2] < z_threshold) & (xyz_world[:, :, 0] >= 0) & (xyz_world[:, :, 1] >= 0))[0], \
                     np.where((xyz_world[:, :, 2] < z_threshold) & (xyz_world[:, :, 1] >= 0) & (xyz_world[:, :, 0] >= 0))[1]
    valid_xyz = xyz_world[x_inds,y_inds]
    valid_xyz = (valid_xyz/res).astype(int)
    valid_xyz[:,0] += grid_centre_x
    valid_xyz[:,1] += grid_centre_y
    # valid_xyz[:, 0][valid_xyz[:,0]>=480] = 479
    # valid_xyz[:, 1][valid_xyz[:, 0]>=640] = 639

    occ_grid_rgb[valid_xyz[:,0],valid_xyz[:,1]] = imc[x_inds,y_inds]

plt.imshow(np.flip(occ_grid_rgb.astype('uint8'),axis=1))