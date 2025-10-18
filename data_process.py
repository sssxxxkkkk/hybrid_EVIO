import os
import tqdm as tqdm
import numpy as np
import copy

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.tools import file_interface, plot
from evo.core.geometry import GeometryException
import evo.main_ape as main_ape
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D

import matplotlib.pyplot as plt #绘图

print("Successfully import ultils")

def make_evo_traj_gt(poses_N_x_7, tss_us):
    assert poses_N_x_7.shape[1] == 7
    assert poses_N_x_7.shape[0] > 10
    assert tss_us.shape[0] == poses_N_x_7.shape[0]

    traj_evo = PoseTrajectory3D(
        positions_xyz=poses_N_x_7[:,:3],
        #orientations_quat_wxyz=poses_N_x_7[:,3:],
        orientations_quat_wxyz = poses_N_x_7[:, [6,3,4,5]],#gt存储的是xyzw
        timestamps=tss_us)#转换为秒
    return traj_evo


def load_gt_us(path, skiprows=0):
    traj_ref = np.loadtxt(path, delimiter=" ", skiprows=skiprows)
    tss_gt_us = traj_ref[1:, 0].copy() 
    assert np.all(tss_gt_us == sorted(tss_gt_us))
    assert traj_ref.shape[0] > 0
    assert traj_ref.shape[1] == 8

    return tss_gt_us, traj_ref[1:, 1:]


def plot_trajectory_inxyplane(pred_traj, gt_traj, align=True, _n_to_align=-1,correct_scale=True, max_diff_sec=1.0,title="", filename=""):
    #将两个轨迹对齐(时间维度上的)
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

    # 对齐轨迹(空间维度上的)
    if align:
        try:
            pred_traj.align(gt_traj, correct_scale=correct_scale,n=_n_to_align)
        except GeometryException as e:
            print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")

    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    
    plot_collection.add_figure("traj (error)", fig)

    plt.show()

print("Successfully define several functions")


def plot_trajectory_rpy(pred_traj, gt_traj, align=True, _n_to_align=-1,correct_scale=True, max_diff_sec=1.0,title="", filename=""):
#将两个轨迹对齐(时间维度上的)
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)
    # 对齐轨迹(空间维度上的)
    if align:
        try:
            pred_traj.align(gt_traj, correct_scale=correct_scale,n=_n_to_align)
        except GeometryException as e:
            print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col",figsize=(8, 8))
    
    plot.traj_rpy(axarr_rpy, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj_rpy(axarr_rpy, pred_traj, '-', 'blue', "Predicted")
    plt.show()


print("Evaluation for DAVIS240c dataset")
mean_ape_err = 0
sequence_num = 0
indir="./"

suffix = "_DL"  # 替换为你需要的后缀
target_dirs = {
                "boxes_6dof",
                "boxes_translation",
                "dynamic_6dof",
                "dynamic_translation",
                "hdr_boxes",
                "hdr_poster",
                "poster_6dof",
                "poster_translation",
                }
target_dirs = {d + suffix for d in target_dirs}

for root, dirs, files in os.walk(indir):
    for d in dirs:
        # 构建完整路径 data_path
        datapath_val = os.path.join(root, d)

        # 检查是否为目标文件夹之一
        if os.path.basename(datapath_val) in target_dirs:
            sequence_name = os.path.basename(datapath_val)

            # 获取轨迹
            tss_gt_us, traj_gt = load_gt_us(os.path.join(datapath_val, f"stamped_groundtruth.txt"))#获取真实轨迹

            # 获取hybrid-估算的轨迹
            tss_deio_us, traj_deio = load_gt_us(os.path.join(datapath_val, f"stamped_traj_estimate.txt"))
            evoGT = make_evo_traj_gt(traj_gt, tss_gt_us)
            evoEst = make_evo_traj_gt(traj_deio, tss_deio_us)
            gtlentraj = evoGT.get_infos()["path length (m)"]#获取轨迹长度
            est_traj = evoEst.get_infos()["path length (m)"]
            print(f"{sequence_name}, est_traj: {est_traj}")
            evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
            _n_to_align=1000;
            ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=False)

            # print(f"\033[31m EVO结果：{ape_trans}\033[0m");
            MPE = ape_trans.stats["mean"] / gtlentraj * 100
            # print(f"MPE is {MPE:.02f}") #注意只保留两位小数
            evoATE = ape_trans.stats["rmse"]*100

            # res_str = f"\nATE[cm]: {evoATE:.02f} | MPE[%/m]: {MPE:.02f}"

            # 添加角度的验证
            ape_rotation = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.rotation_angle_deg, align=True,n_to_align=_n_to_align, correct_scale=False)
            MRE = ape_rotation.stats["mean"]/ gtlentraj #以度为单位的均值
            
            sequence_num += 1
            mean_ape_err += MPE 

            rmse_degree = ape_rotation.stats["rmse"]
            res_str = f"\nATE[cm]: {evoATE:.02f} | MPE[%/m]: {MPE:.02f}  | rmse_degree[deg]: {rmse_degree:.02f} | MRE[deg/m]: {MRE:.02f}"

            #matplotlib inline
            plot_trajectory_rpy(copy.deepcopy(evoGT), copy.deepcopy(evoEst),_n_to_align=_n_to_align);
            
            print(f"{sequence_name}: {res_str}")
            
    # 使用break限制os.walk只遍历indir的第一层
    break

print(f"Mean MPE[%]: {mean_ape_err/sequence_num:.02f}")


print("Evaluation for MONO-HKU dataset")
sequence_num = 0
mean_ape_err = 0

target_dirs = {
                "vicon_dark1",
                "vicon_dark2",
                "vicon_darktolight1",
                "vicon_darktolight2",
                "vicon_hdr1",
                "vicon_hdr2",
                "vicon_hdr3",
                "vicon_hdr4",
                "vicon_lighttodark1",
                "vicon_lighttodark2",
                }
target_dirs = {d + suffix for d in target_dirs}


for root, dirs, files in os.walk(indir):
    for d in dirs:
        # 构建完整路径 data_path
        datapath_val = os.path.join(root, d)

        # 检查是否为目标文件夹之一
        if os.path.basename(datapath_val) in target_dirs:
            sequence_name = os.path.basename(datapath_val)

            # 获取轨迹
            tss_gt_us, traj_gt = load_gt_us(os.path.join(datapath_val, f"stamped_groundtruth.txt"))#获取真实轨迹

            # 获取hybrid-evio估算的轨迹
            tss_deio_us, traj_deio = load_gt_us(os.path.join(datapath_val, f"stamped_traj_estimate.txt")) 

            evoGT = make_evo_traj_gt(traj_gt, tss_gt_us)
            evoEst = make_evo_traj_gt(traj_deio, tss_deio_us)
            gtlentraj = evoGT.get_infos()["path length (m)"]#获取轨迹长度
            evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
            _n_to_align=251;
            ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=False)

            # print(f"\033[31m EVO结果：{ape_trans}\033[0m");
            MPE = ape_trans.stats["mean"] / gtlentraj * 100
            # print(f"MPE is {MPE:.02f}") #注意只保留两位小数
            evoATE = ape_trans.stats["rmse"]*100

            res_str = f"\nATE[cm]: {evoATE:.02f} | MPE[%/m]: {MPE:.02f}"
            
            sequence_num += 1
            mean_ape_err += MPE

            print(f"{sequence_name}: {res_str}")
            
    # 使用break限制os.walk只遍历indir的第一层
    break
print(f"Mean MPE[%]: {mean_ape_err/sequence_num:.02f}")

print("Evaluation for UZH-FPV dataset")
mean_ape_err = 0
sequence_num = 0
target_dirs = {
                "indoor_45_2_davis_with_gt",
                "indoor_45_4_davis_with_gt",
                "indoor_45_9_davis_with_gt",
                "indoor_forward_3_davis_with_gt",
                "indoor_forward_5_davis_with_gt",
                "indoor_forward_6_davis_with_gt",
                "indoor_forward_7_davis_with_gt",
                "indoor_forward_9_davis_with_gt",
                "indoor_forward_10_davis_with_gt",
                }
target_dirs = {d + suffix for d in target_dirs}

for root, dirs, files in os.walk(indir):
    for d in dirs:
        # 构建完整路径 data_path
        datapath_val = os.path.join(root, d)

        # 检查是否为目标文件夹之一
        if os.path.basename(datapath_val) in target_dirs:
            sequence_name = os.path.basename(datapath_val)

            # 获取轨迹
            tss_gt_us, traj_gt = load_gt_us(os.path.join(datapath_val, f"stamped_groundtruth.txt"))#获取真实轨迹

            # 获取hybrid-evio估算的轨迹
            tss_deio_us, traj_deio = load_gt_us(os.path.join(datapath_val, f"stamped_traj_estimate.txt")) 

            evoGT = make_evo_traj_gt(traj_gt, tss_gt_us)
            evoEst = make_evo_traj_gt(traj_deio, tss_deio_us)
            gtlentraj = evoGT.get_infos()["path length (m)"]#获取轨迹长度
            evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
            _n_to_align=-1;
            ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=False)

            # print(f"\033[31m EVO结果：{ape_trans}\033[0m");
            MPE = ape_trans.stats["mean"] / gtlentraj * 100
            # print(f"MPE is {MPE:.02f}") #注意只保留两位小数
            evoATE = ape_trans.stats["rmse"]*100

            res_str = f"\nATE[cm]: {evoATE:.02f} | MPE[%/m]: {MPE:.02f}"

            sequence_num += 1
            mean_ape_err += MPE      

            print(f"{sequence_name}: {res_str}")
            
    # 使用break限制os.walk只遍历indir的第一层
    break

print(f"Mean MPE[%]: {mean_ape_err/sequence_num:.02f}")



print("Evaluation for EDS dataset")
sequence_num = 0
mpe_err = 0
ate_err = 0

target_dirs = {
                "peanuts_dark",
                "peanuts_light",
                "rocket_dark",
                "ziggy_flying_pieces",
                }
target_dirs = {d + suffix for d in target_dirs}


for root, dirs, files in os.walk(indir):
    for d in dirs:
        # 构建完整路径 data_path
        datapath_val = os.path.join(root, d)

        # 检查是否为目标文件夹之一
        if os.path.basename(datapath_val) in target_dirs:
            sequence_name = os.path.basename(datapath_val)

            # 获取轨迹
            tss_gt_us, traj_gt = load_gt_us(os.path.join(datapath_val, f"stamped_groundtruth.txt"))#获取真实轨迹

            # 获取hybrid-evio估算的轨迹
            tss_deio_us, traj_deio = load_gt_us(os.path.join(datapath_val, f"stamped_traj_estimate.txt")) 

            evoGT = make_evo_traj_gt(traj_gt, tss_gt_us)
            evoEst = make_evo_traj_gt(traj_deio, tss_deio_us)
            gtlentraj = evoGT.get_infos()["path length (m)"]#获取轨迹长度
            evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
            _n_to_align=-1;
            ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=True)

            # print(f"\033[31m EVO结果：{ape_trans}\033[0m");
            MPE = ape_trans.stats["mean"] / gtlentraj * 100
            # print(f"MPE is {MPE:.02f}") #注意只保留两位小数
            evoATE = ape_trans.stats["rmse"]*100

            res_str = f"\nATE[cm]: {evoATE:.02f} | MPE[%/m]: {MPE:.02f}"
            
            sequence_num += 1
            mpe_err += MPE
            ate_err += evoATE

            print(f"{sequence_name}: {res_str}")
            
    # 使用break限制os.walk只遍历indir的第一层
    break
print(f"Mean ATE[cm]: {ate_err/sequence_num:.02f}")    
print(f"Mean MPE[%]: {mpe_err/sequence_num:.02f}")
