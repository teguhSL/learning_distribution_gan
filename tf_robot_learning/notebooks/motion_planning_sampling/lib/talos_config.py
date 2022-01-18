import numpy as np

#pb_joint_indices = np.array(list(np.arange(45,51)) + list(np.arange(52,58)) + [0, 1] + \
#                list(np.arange(11,18)) +[21] + list(np.arange(28,35)) + [38,3,4]).astype(int)
# pb_joint_indices = np.array([45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57,  0,  1, 11, 12, 13,
#        14, 15, 16, 17, 28, 29, 30, 31, 32, 33, 34])
tf_joint_indices = np.array([ 0,  1, 28, 29, 30, 31, 32, 33, 34, 11, 12, 13, 14, 15, 16, 17, 52, 53, 54, 55, 56, 57, 45, 46, 47, 48, 49, 50])#r 'right, left'
pb_joint_names_complete = ['torso_1_joint', 'torso_2_joint', 'imu_joint',
                           'head_1_joint', 'head_2_joint', 'rgbd_joint',
                           'rgbd_optical_joint', 'rgbd_depth_joint',
                           'rgbd_depth_optical_joint', 'rgbd_rgb_joint',
                           'rgbd_rgb_optical_joint', 'arm_left_1_joint',
                           'arm_left_2_joint', 'arm_left_3_joint',
                           'arm_left_4_joint', 'arm_left_5_joint',
                           'arm_left_6_joint', 'arm_left_7_joint',
                           'wrist_left_ft_joint', 'wrist_left_tool_joint',
                           'gripper_left_base_link_joint', 'gripper_left_joint',
                           'gripper_left_inner_double_joint',
                           'gripper_left_fingertip_1_joint',
                           'gripper_left_fingertip_2_joint',
                           'gripper_left_motor_single_joint',
                           'gripper_left_inner_single_joint',
                           'gripper_left_fingertip_3_joint', 'arm_right_1_joint',
                           'arm_right_2_joint', 'arm_right_3_joint',
                           'arm_right_4_joint', 'arm_right_5_joint',
                           'arm_right_6_joint', 'arm_right_7_joint',
                           'wrist_right_ft_joint', 'wrist_right_tool_joint',
                           'gripper_right_base_link_joint', 'gripper_right_joint',
                           'gripper_right_inner_double_joint',
                           'gripper_right_fingertip_1_joint',
                           'gripper_right_fingertip_2_joint',
                           'gripper_right_motor_single_joint',
                           'gripper_right_inner_single_joint',
                           'gripper_right_fingertip_3_joint',
                           'leg_left_1_joint', 'leg_left_2_joint',
                           'leg_left_3_joint', 'leg_left_4_joint',
                           'leg_left_5_joint', 'leg_left_6_joint',
                           'leg_left_sole_fix_joint', 'leg_right_1_joint',
                           'leg_right_2_joint', 'leg_right_3_joint',
                           'leg_right_4_joint', 'leg_right_5_joint',
                           'leg_right_6_joint', 'leg_right_sole_fix_joint']

q0Complete =  np.array([ 0.     , 0.    ,  1.019272,  0.      ,  0.      ,  0.,
        1., 0.00000e+00 , 0.00000e+00, -4.11354e-01 , 8.59395e-01 ,-4.48041e-01,
  -1.70800e-03 , 0.00000e+00 , 0.00000e+00 ,-4.11354e-01 , 8.59395e-01,
  -4.48041e-01 ,-1.70800e-03 , 0.00000e+00 , 6.76100e-03,  2.58470e-01,
   1.73046e-01 ,-2.00000e-04 ,-5.25366e-01 , 0.00000e+00 , 0.00000e+00,
   1.00000e-01 , 1.00000e-01 ,-1.73046e-01 , 2.00000e-04,
  -5.25366e-01 , 0.00000e+00 , 0.00000e+00 , 1.00000e-01 ])


def get_pb_config(q):
    """
    Convert tf's format of 'joint angles + base position' to
    'base_pos + base_ori + joint_angles' according to pybullet order
    """
    joint_angles = q[:28]
    # qnew = np.concatenate([q[28:31], euler2quat(q[-3:]),
    # qnew = np.concatenate([np.array([0,0,q[28]]), euler2quat(q[-3:]),
    qnew = np.concatenate([q[28:31], euler2quat(np.array([0, 0, 0])),
                           joint_angles[-6:], joint_angles[-12:-6],
                           joint_angles[:2], joint_angles[9:16],
                           joint_angles[2:9]])
    return qnew


def get_tf_config(q):
    """
    Convert 'base_pos + base_ori + joint_angles' according to pybullet order
    to tf's format of 'joint angles + base position'

    """
    joint_angles = q[7:]
    qnew = np.concatenate([joint_angles[12:14], joint_angles[-7:],
                           joint_angles[-14:-7], joint_angles[6:12],
                           joint_angles[:6], q[:3]])
    return qnew