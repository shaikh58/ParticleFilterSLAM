import numpy as np
from preprocess import *
from scipy.linalg import expm
from configs import *
import matplotlib.pyplot as plt
import time
from scipy.special import expit
import matplotlib

N = 20
# COVAR_NOISE = np.zeros((3,3))
# COVAR_NOISE[0,0] = SIGMA_V
# COVAR_NOISE[1,1] = SIGMA_V
# COVAR_NOISE[2,2] = SIGMA_W
OCCUPIED_THRESHOLD = 0.92

def nonvec_motion_model(T,tau,w,v):
    '''takes in 3x3 pose matrix T and returns prediction for single pose'''
    twist_hat = np.zeros((3,3))
    twist_hat[1,0] = w
    twist_hat[0,1] = -w
    twist_hat[0,2] = v
    return T @ expm(tau * twist_hat)

# initial occupancy grid map
arr_poses = np.zeros((n_enc,3,3))
arr_poses[0,0:2,0:2] = np.eye(2)
arr_poses[:,2,2] = 1
# transform initial (t=0) lidar scan to world frame from lidar frame
wTl = np.matmul(arr_poses[0],bTl)
xy_lidar_hom = np.vstack((x_lidar[:,0], y_lidar[:,0], np.ones((x_lidar.shape[0],))))
lidar_scan_w_0 = np.matmul(wTl,xy_lidar_hom)
# create the occupancy grid matrix; centre of matrix (between 400,400) is world origin
OCCUPANCY_GRID_SIZE_INIT = (800,800)
occ_grid_init = np.ones(OCCUPANCY_GRID_SIZE_INIT)
grid_centre_x_init = int(OCCUPANCY_GRID_SIZE_INIT[0]/2 - 1)
grid_centre_y_init = int(OCCUPANCY_GRID_SIZE_INIT[0]/2 - 1)

# for i in range(lidar_scan_w_0.shape[1]):
#     # initial lidar scan
#     filled_pixels = bresenham2D(0+LIDAR_X_OFFSET,0,lidar_scan_w_0[0,i]/res,lidar_scan_w_0[1,i]/res)
#     occ_grid_init[grid_centre_x_init + filled_pixels[0].astype(int), grid_centre_y_init - filled_pixels[1].astype(int)] = 0
# plt.imshow(occ_grid_init.T)

##############################################
# dead reckoning with complete occupancy grid
start = time.time()
arr_lidar_xy_world = np.zeros((n_enc,3,1081))
# occ_grid_deadreck = np.ones(OCCUPANCY_GRID_SIZE_INIT)
for i in range(n_enc-1):
    # if i%10==0:
        # print(i)
    arr_poses[i+1] = nonvec_motion_model(arr_poses[i],encoder_stamps[i+1] - encoder_stamps[i], reduced_imu[i],v_lin[i])
    # transform each lidar scan x,y pair to world x,y and fill into the occupancy grid
    xy_lidar_hom_ip1 = np.vstack((x_lidar[:, i+1], y_lidar[:, i+1], np.ones((x_lidar.shape[0],))))
    # # recall to convert lidar to world we need body to world and lidar to body
    arr_lidar_xy_world[i+1] = np.matmul(arr_poses[i+1],bTl) @ xy_lidar_hom_ip1
    # # ray tracing for each x,y pair in lidar scan to find filled in cells
#     for p_ix in range(arr_lidar_xy_world[i+1].shape[1]):
#         if np.isnan(arr_lidar_xy_world[i+1,0,p_ix]) or np.isnan(arr_lidar_xy_world[i+1,1,p_ix]):
#             pass
#         else:
#             filled_pixels = bresenham2D(np.matmul(arr_poses[i+1],bTl)[0,2]*LIDAR_SCALE_FACTOR,
#                                         np.matmul(arr_poses[i+1],bTl)[1,2]*LIDAR_SCALE_FACTOR,
#                                         arr_lidar_xy_world[i+1,0,p_ix]*LIDAR_SCALE_FACTOR,
#                                         arr_lidar_xy_world[i+1,1,p_ix]*LIDAR_SCALE_FACTOR)
#             # update occupancy grid with filled in cells
#             occ_grid_deadreck[grid_centre_x + filled_pixels[0].astype(int), grid_centre_y - filled_pixels[1].astype(int)] = 0
# print('Time taken: ', time.time() - start)
# plt.imshow(occ_grid_deadreck.T, cmap = 'Greys')

# plot dead reckoning trajectory
# arr_x = arr_poses[:,0,2]
# arr_y = arr_poses[:,1,2]
# plt.plot(arr_x,arr_y)

# particle filter SLAM

# calculate the max x and min in metres for the map correlation (use dead reckoning)
xmin = int(np.ceil(arr_lidar_xy_world[:,0,:].min()/res) - 10/res)
xmax = int(np.ceil(arr_lidar_xy_world[:,0,:].max()/res) + 10/res)
ymin = int(np.ceil(arr_lidar_xy_world[:,1,:].min()/res) - 10/res)
ymax = int(np.ceil(arr_lidar_xy_world[:,1,:].max()/res) + 10/res)
OCCUPANCY_GRID_SIZE_DIM = int(np.ceil(max(int((xmax-xmin) + 1),int((ymax-ymin) + 1))/10)*10)
OCCUPANCY_GRID_SIZE = (OCCUPANCY_GRID_SIZE_DIM,OCCUPANCY_GRID_SIZE_DIM)
grid_centre_x = int(OCCUPANCY_GRID_SIZE[0]/2 - 1)
grid_centre_y = int(OCCUPANCY_GRID_SIZE[0]/2 - 1)

# initialize the occupancy grid with log odds from the initial lidar scan
start = time.time()
occ_grid = np.zeros(OCCUPANCY_GRID_SIZE)
for i in range(lidar_scan_w_0.shape[1]):
    # initial lidar scan
    filled_pixels = bresenham2D(0+LIDAR_X_OFFSET,0,lidar_scan_w_0[0,i]/res,lidar_scan_w_0[1,i]/res)
    occ_grid[grid_centre_x + filled_pixels[0,:-1].astype(int), grid_centre_y + filled_pixels[1,:-1].astype(int)] -= np.log(4)
    occ_grid[
        grid_centre_x + filled_pixels[0, -1].astype(int), grid_centre_y + filled_pixels[1, -1].astype(int)] += np.log(4)
# ONLY FOR PLOTTING PURPOSES: apply sigmoid function to turn into probabilities then threshold and plot
occ_grid_sigmoid = expit(occ_grid)
# plt.imshow(np.flip(occ_grid_sigmoid,axis=1).T)
# main loop
# initialize N particles poses at (0,0,0)
particles = np.zeros((x_lidar.shape[1],N,3,3))
particles[0,:,0:2,0:2] = np.eye(2)
particles[:,:,2,2] = 1
alphas = np.zeros((x_lidar.shape[1],N,))
alphas[0,:] = 1/N
map_corrs = np.zeros((x_lidar.shape[1],N,9,9))
arr_lidar_xy_world_particle = np.zeros((x_lidar.shape[1],N,3,1081))
arr_lidar_xy_world_particle[0,:,:,:] = np.vstack((x_lidar[:, 0], y_lidar[:, 0], np.ones((x_lidar.shape[0],))))
arr_correlation_part = np.zeros((x_lidar.shape[1],N))
# loop through every time stamp
for i in range(x_lidar.shape[1] - 1):
    # get x,y coords of lidar scan in lidar frame
    xy_lidar_hom_part_i = np.vstack((x_lidar[:, i+1], y_lidar[:, i+1], np.ones((x_lidar.shape[0],))))
    for part in range(N):
        # prediction step (motion model) with noise
        # add noise relative to the scale of the velocity and angular velocity
        particles[i+1,part] = nonvec_motion_model(particles[i,part,:,:], encoder_stamps[i+1] - encoder_stamps[i],
                                                  reduced_imu[i]+np.random.normal(0,np.abs(reduced_imu[i]*VAR_SCALE_FACTOR)),
                                                  v_lin[i]+np.random.normal(0,np.abs(v_lin[i]*VAR_SCALE_FACTOR)))
        # to convert lidar scan x,y coords to world we need body to world and lidar to body
        arr_lidar_xy_world_particle[i+1,part] = np.matmul(particles[i+1,part], bTl) @ xy_lidar_hom_part_i
    ###########################
    # # calculate map correlation for each particle (create arguments for map corr function in pr2_utils.py)
    # x_im = np.arange(xmin, xmax + res, res)  # x-positions of each pixel of the map
    # y_im = np.arange(ymin, ymax + res, res)  # y-positions of each pixel of the map
    # # loop over each particle and calculate map correlation for each
    # arr_weight_max_particle_i = np.zeros((N,))
    for p in range(N):
        ####### map correlation implementation ##########
        # create occupancy matrix for each lidar scan for each particle - reset each time
        lidar_xy_map_frame_occ_grid = np.zeros(OCCUPANCY_GRID_SIZE)
        im = (occ_grid_sigmoid > OCCUPIED_THRESHOLD).astype(int)
        Y = np.round(
            arr_lidar_xy_world_particle[i + 1, p] / res)  # scaled lidar scan endpoint x,y coords in world frame
        # check that the coordinates are within the map boundaries
        map_x_posn_arr = (grid_centre_x + Y[0]).astype(int)
        map_y_posn_arr = (grid_centre_y + Y[1]).astype(int)
        map_x_posn_arr[map_x_posn_arr >= OCCUPANCY_GRID_SIZE[0]] = OCCUPANCY_GRID_SIZE[0] - 1
        map_y_posn_arr[map_y_posn_arr >= OCCUPANCY_GRID_SIZE[0]] = OCCUPANCY_GRID_SIZE[0] - 1
        # args_x_inside_map = np.where(map_x_posn_arr < OCCUPANCY_GRID_SIZE[0])[0]
        # args_y_inside_map = np.where(map_y_posn_arr < OCCUPANCY_GRID_SIZE[0])[0]
        # limiting_axis = np.argmin(np.array([args_x_inside_map.shape[0],args_y_inside_map.shape[0]]))
        #
        # update_x_args_map = map_x_posn_arr[args_x_inside_map]
        # update_y_args_map = map_y_posn_arr[args_y_inside_map]
        lidar_xy_map_frame_occ_grid[map_x_posn_arr, map_y_posn_arr] = 1
        # plt.imshow(lidar_xy_map_frame_occ_grid)
        # compute correlation i.e. number of matching entries in the 2 occupancy grids
        correlation = np.sum(lidar_xy_map_frame_occ_grid == im) + 1
        # print('Correlation %: ', correlation/OCCUPANCY_GRID_SIZE[0]**2)
        arr_correlation_part[i+1,p] = correlation*alphas[i,p]


    # update particle weights
    arr_weight_max_particle_i = arr_correlation_part[i+1]
    alphas[i+1,:] = arr_weight_max_particle_i/np.sum(arr_weight_max_particle_i)
    argmax_updated_weight_normed = np.argmax(alphas[i+1,:])
    # choose particle with highest weight (assume this is where robot is for map update)
    map_update_optimal_particle = particles[i+1, argmax_updated_weight_normed]
    # plot the x,y coordinates of the particle with the highest weight
    plt.scatter(map_update_optimal_particle[0,2],map_update_optimal_particle[1,2],s=1,color='black')
    # convert lidar scan x,y coords to world assuming the optimal particle is the true robot pose
    lidar_xy_world_opt_particle = np.matmul(map_update_optimal_particle, bTl) @ xy_lidar_hom_part_i
    # update map log odds with bresenham; loop through each reading in the lidar scan for time i+1
    for r in range(arr_lidar_xy_world_particle[i + 1].shape[-1]):
        try:
            if np.isnan(arr_lidar_xy_world_particle[i+1,argmax_updated_weight_normed,0,r]) or np.isnan(arr_lidar_xy_world_particle[i+1,argmax_updated_weight_normed,1,r]):
                pass
            else:
                # scale the pose coords and the lidar scan coords
                # bres_pixels = bresenham2D(np.matmul(particles[i+1,0], bTl)[0, 2] / res,
                #                           np.matmul(particles[i+1,0], bTl)[1, 2] / res,
                #                           arr_lidar_xy_world_particle[i + 1, 0, 0, r] / res,
                #                           arr_lidar_xy_world_particle[i + 1, 0, 1, r] / res)
                bres_pixels = bresenham2D(np.matmul(map_update_optimal_particle,bTl)[0,2]/res,
                                          np.matmul(map_update_optimal_particle,bTl)[1,2]/res,
                                          arr_lidar_xy_world_particle[i+1,argmax_updated_weight_normed,0,r]/res,
                                          arr_lidar_xy_world_particle[i+1,argmax_updated_weight_normed,1,r]/res)
                # update probabilistic occupancy map; reduce log odds of free cells, increase log odds of occupied cells
                # occ_grid[grid_centre_x + bres_pixels[0].astype(int), grid_centre_y + bres_pixels[1].astype(
                #         int)] = 0
                occ_grid[grid_centre_x + bres_pixels[0, :-1].astype(int), grid_centre_y + bres_pixels[1, :-1].astype(
                    int)] -= np.log(4)
                occ_grid[
                    grid_centre_x + bres_pixels[0, -1].astype(int), grid_centre_y + bres_pixels[1, -1].astype(
                        int)] += np.log(4)
        except Exception as e:
            print(e, ' - skipping lidar scan {x}'.format(x=r))
            pass
    # check if resampling needs to be done - slide 8
    N_eff = 1/(np.sum(alphas[i+1,:]**2))
    if i % 100 == 0:
        print(i)
        print('Updated weights: ', alphas[i+1,:])
        print('Optimal particle #: ', argmax_updated_weight_normed)
        print('Effective # of particles: ', N_eff)
        print('time so far: ', time.time() - start)
    if N_eff <= N/10:
        print('Particles resampled!')
        # resample in this case; resample according to previous weights and set to uniform weight
        resampled_pose_indices = np.random.choice(N,N,p=alphas[i+1,:])
        particles[i+1] = particles[i,resampled_pose_indices]
        alphas[i+1,:] = 1/N

    occ_grid_sigmoid = expit(occ_grid)

# plt.show()
# plt.plot(ix,iy)
plt.imshow(np.flip(occ_grid_sigmoid,axis=1).T)
# plt.imsave(fname='../plots/d21_map.png',arr=np.flip(occ_grid_sigmoid,axis=1).T,dpi=600)
print('time taken: ', time.time() - start)

# arr_x_part = particles[:,:,0,2]
# arr_y_part = particles[:,:,1,2]
# plt.plot(arr_x_part,arr_y_part)
# plt.title('Particle filter trajectory for N=20 particles')

# np.save('../data/d21_particle_filter_poses_n=20',particles)
# np.save('../data/d21_particle_filter_trajectory_x_coords_n=20',arr_x_part)
# np.save('../data/d21_particle_filter_trajectory_y_coords_n=20',arr_y_part)
# np.save('../data/particle_filter_occ_grid_threshold_dataset21',(occ_grid_sigmoid > OCCUPIED_THRESHOLD).astype(int))