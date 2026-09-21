_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
randomness = dict(seed=1, deterministic=True, diff_rank_seed=False)

