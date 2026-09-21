# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
# mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py
# configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py
# configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py
# configs/attention/second.py
# configs/mvxnet/points_voxel_fusion.py12
# configs/mvxnet/pointpainting.py
# /liaopeng/pointcode/mmdetectiond-mai/configs/mvxnet/popython2
# <local-path>/point_code/mmdetection3d-main/configs/mvxnet/points_voxel_fusion_cc_car.py
# <local-path>/point_code/mmdetection3d-main/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-car.py
# <local-path>/point_code/mmdetection3d-main/configs/pv_rcnn/pv_rcnn_org_test_per.py
def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', help='train config file path',
                        default='<local-path>/point_code/mmdetection3d-main-nus/configs(ab)/mvxnet/points_voxel_fusion_cc.py')
    # mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class pointpainting configs/mvxnet/pointpainting.py
    # parser.add_argument('--config', help='train config file path',
    #                     default='../configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
             '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():

    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    # python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()

