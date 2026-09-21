# 鑷畾涔夋ā鍨?
鎴戜滑閫氬父鎶婃ā鍨嬬殑鍚勪釜缁勬垚鎴愬垎鍒嗘垚 6 绉嶇被鍨嬶細

- 缂栫爜鍣紙encoder锛夛細鍖呮嫭 voxel encoder 鍜?middle encoder 绛夎繘鍏?backbone 鍓嶆墍浣跨敤鐨勫熀浜庝綋绱犵殑鏂规硶锛屽 `HardVFE` 鍜?`PointPillarsScatter`銆?- 楠ㄥ共缃戠粶锛坆ackbone锛夛細閫氬父閲囩敤 FCN 缃戠粶鏉ユ彁鍙栫壒寰佸浘锛屽 `ResNet` 鍜?`SECOND`銆?- 棰堥儴缃戠粶锛坣eck锛夛細浣嶄簬 backbones 鍜?heads 涔嬮棿鐨勭粍鎴愭ā鍧楋紝濡?`FPN` 鍜?`SECONDFPN`銆?- 妫€娴嬪ご锛坔ead锛夛細鐢ㄤ簬鐗瑰畾浠诲姟鐨勭粍鎴愭ā鍧楋紝濡俙妫€娴嬫鐨勯娴媊鍜宍鎺╃爜鐨勯娴媊銆?- RoI 鎻愬彇鍣紙RoI extractor锛夛細鐢ㄤ簬浠庣壒寰佸浘涓彁鍙?RoI 鐗瑰緛鐨勭粍鎴愭ā鍧楋紝濡?`H3DRoIHead` 鍜?`PartAggregationROIHead`銆?- 鎹熷け鍑芥暟锛坙oss锛夛細heads 涓敤浜庤绠楁崯澶卞嚱鏁扮殑缁勬垚妯″潡锛屽 `FocalLoss`銆乣L1Loss` 鍜?`GHMLoss`銆?
## 寮€鍙戞柊鐨勭粍鎴愭ā鍧?
### 娣诲姞鏂扮殑缂栫爜鍣?
鎺ヤ笅鏉ユ垜浠互 HardVFE 涓轰緥灞曠ず濡備綍寮€鍙戞柊鐨勭粍鎴愭ā鍧椼€?
#### 1. 瀹氫箟涓€涓柊鐨勪綋绱犵紪鐮佸櫒锛堝 HardVFE锛氬嵆 HV-SECOND 涓娇鐢ㄧ殑浣撶礌鐗瑰緛缂栫爜鍣級

鍒涘缓涓€涓柊鏂囦欢 `mmdet3d/models/voxel_encoders/voxel_encoder.py`銆?
```python
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class HardVFE(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # 闇€瑕佽繑鍥炰竴涓厓缁?        pass
```

#### 2. 瀵煎叆璇ユā鍧?
鎮ㄥ彲浠ュ湪 `mmdet3d/models/voxel_encoders/__init__.py` 涓坊鍔犱互涓嬩唬鐮侊細

```python
from .voxel_encoder import HardVFE
```

鎴栬€呭湪閰嶇疆鏂囦欢涓坊鍔犱互涓嬩唬鐮侊紝浠庤€岄伩鍏嶄慨鏀规簮鐮侊細

```python
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.voxel_encoder'],
    allow_failed_imports=False)
```

#### 3. 鍦ㄩ厤缃枃浠朵腑浣跨敤浣撶礌缂栫爜鍣?
```python
model = dict(
    ...
    voxel_encoder=dict(
        type='HardVFE',
        arg1=xxx,
        arg2=yyy),
    ...
)
```

### 娣诲姞鏂扮殑楠ㄥ共缃戠粶

鎺ヤ笅鏉ユ垜浠互 [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)锛圫parsely Embedded Convolutional Detection锛変负渚嬪睍绀哄浣曞紑鍙戞柊鐨勭粍鎴愭ā鍧椼€?
#### 1. 瀹氫箟涓€涓柊鐨勯骞茬綉缁滐紙濡?SECOND锛?
鍒涘缓涓€涓柊鏂囦欢 `mmdet3d/models/backbones/second.py`銆?
```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SECOND(BaseModule):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # 闇€瑕佽繑鍥炰竴涓厓缁?        pass
```

#### 2. 瀵煎叆璇ユā鍧?
鎮ㄥ彲浠ュ湪 `mmdet3d/models/backbones/__init__.py` 涓坊鍔犱互涓嬩唬鐮侊細

```python
from .second import SECOND
```

鎴栬€呭湪閰嶇疆鏂囦欢涓坊鍔犱互涓嬩唬鐮侊紝浠庤€岄伩鍏嶄慨鏀规簮鐮侊細

```python
custom_imports = dict(
    imports=['mmdet3d.models.backbones.second'],
    allow_failed_imports=False)
```

#### 3. 鍦ㄩ厤缃枃浠朵腑浣跨敤楠ㄥ共缃戠粶

```python
model = dict(
    ...
    backbone=dict(
        type='SECOND',
        arg1=xxx,
        arg2=yyy),
    ...
)
```

### 娣诲姞鏂扮殑棰堥儴缃戠粶

#### 1. 瀹氫箟涓€涓柊鐨勯閮ㄧ綉缁滐紙濡?SECONDFPN锛?
鍒涘缓涓€涓柊鏂囦欢 `mmdet3d/models/necks/second_fpn.py`銆?
```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SECONDFPN(BaseModule):

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        pass

    def forward(self, x):
        # 鍏蜂綋瀹炵幇蹇界暐
        pass
```

#### 2. 瀵煎叆璇ユā鍧?
鎮ㄥ彲浠ュ湪 `mmdet3d/models/necks/__init__.py` 涓坊鍔犱互涓嬩唬鐮侊細

```python
from .second_fpn import SECONDFPN
```

鎴栬€呭湪閰嶇疆鏂囦欢涓坊鍔犱互涓嬩唬鐮侊紝浠庤€岄伩鍏嶄慨鏀规簮鐮侊細

```python
custom_imports = dict(
    imports=['mmdet3d.models.necks.second_fpn'],
    allow_failed_imports=False)
```

#### 3. 鍦ㄩ厤缃枃浠朵腑浣跨敤棰堥儴缃戠粶

```python
model = dict(
    ...
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    ...
)
```

### 娣诲姞鏂扮殑妫€娴嬪ご

鎺ヤ笅鏉ユ垜浠互 [PartA2 Head](https://arxiv.org/abs/1907.03670) 涓轰緥灞曠ず濡備綍寮€鍙戞柊鐨勬娴嬪ご銆?
**娉ㄦ剰**锛氭澶勫睍绀虹殑 `PartA2 RoI Head` 灏嗙敤浜庢娴嬪櫒鐨勭浜岄樁娈点€傚浜庡崟闃舵鐨勬娴嬪ご锛岃鍙傝€?`mmdet3d/models/dense_heads/` 涓殑渚嬪瓙銆傜敱浜庡叾绠€鍗曢珮鏁堬紝瀹冧滑鏇村父鐢ㄤ簬鑷姩椹鹃┒鍦烘櫙涓嬬殑 3D 妫€娴嬩腑銆?
棣栧厛锛屽湪 `mmdet3d/models/roi_heads/bbox_heads/parta2_bbox_head.py` 涓坊鍔犳柊鐨?bbox head銆俙PartA2 RoI Head` 涓虹洰鏍囨娴嬪疄鐜颁簡涓€涓柊鐨?bbox head銆備负浜嗗疄鐜颁竴涓?bbox head锛屾垜浠€氬父闇€瑕佸湪鏂版ā鍧椾腑瀹炵幇濡備笅涓や釜鍑芥暟銆傛湁鏃惰繕闇€瑕佸疄鐜板叾浠栫浉鍏冲嚱鏁帮紝濡?`loss` 鍜?`get_targets`銆?
```python
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class PartA2BboxHead(BaseModule):
    """PartA2 RoI head."""

    def __init__(self,
                 num_classes,
                 seg_in_channels,
                 part_in_channels,
                 seg_conv_channels=None,
                 part_conv_channels=None,
                 merge_conv_channels=None,
                 down_conv_channels=None,
                 shared_fc_channels=None,
                 cls_channels=None,
                 reg_channels=None,
                 dropout_ratio=0.1,
                 roi_feat_size=14,
                 with_corner_loss=True,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(PartA2BboxHead, self).__init__(init_cfg=init_cfg)

    def forward(self, seg_feats, part_feats):
        pass
```

鍏舵锛屽鏋滄湁蹇呰鐨勮瘽闇€瑕佸疄鐜颁竴涓柊鐨?RoI Head銆傛垜浠粠 `Base3DRoIHead` 涓户鎵垮緱鍒版柊鐨?`PartAggregationROIHead`銆傛垜浠彲浠ュ彂鐜?`Base3DRoIHead` 宸茬粡瀹炵幇浜嗗涓嬪嚱鏁般€?
```python
from mmdet.models.roi_heads import BaseRoIHead

from mmdet3d.registry import MODELS, TASK_UTILS


class Base3DRoIHead(BaseRoIHead):
    """Base class for 3d RoIHeads."""

    def __init__(self,
                 bbox_head=None,
                 bbox_roi_extractor=None,
                 mask_head=None,
                 mask_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Base3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_head=mask_head,
            mask_roi_extractor=mask_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    TASK_UTILS.build(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)

    def init_mask_head(self):
        """Initialize mask head, skip since ``PartAggregationROIHead`` does not
        have one."""
        pass
```

鎺ヤ笅鏉ヤ富瑕佸 bbox_forward 鐨勯€昏緫杩涜淇敼锛屽悓鏃跺叾缁ф壙浜嗘潵鑷?`Base3DRoIHead` 鐨勫叾瀹冮€昏緫銆傚湪 `mmdet3d/models/roi_heads/part_aggregation_roi_head.py` 涓紝鎴戜滑瀹炵幇浜嗘柊鐨?RoI Head锛屽涓嬫墍绀猴細

```python
from typing import Dict, List, Tuple

from mmdet.models.task_modules import AssignResult, SamplingResult
from mmengine import ConfigDict
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.utils import InstanceList
from ...structures.det3d_data_sample import SampleList
from .base_3droi_head import Base3DRoIHead


@MODELS.register_module()
class PartAggregationROIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        bbox_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 semantic_head: dict,
                 num_classes: int = 3,
                 seg_roi_extractor: dict = None,
                 bbox_head: dict = None,
                 bbox_roi_extractor: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(PartAggregationROIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        assert semantic_head is not None
        self.init_seg_head(seg_roi_extractor, semantic_head)

    def init_seg_head(self, seg_roi_extractor: dict,
                      semantic_head: dict) -> None:
        """Initialize semantic head and seg roi extractor.

        Args:
            seg_roi_extractor (dict): Config of seg
                roi extractor.
            semantic_head (dict): Config of semantic head.
        """
        self.semantic_head = MODELS.build(semantic_head)
        self.seg_roi_extractor = MODELS.build(seg_roi_extractor)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def predict(self,
                feats_dict: Dict,
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        assert self.with_bbox, 'Bbox head must be implemented in PartA2.'
        assert self.with_semantic, 'Semantic head must be implemented' \
                                   ' in PartA2.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        voxels_dict = feats_dict.pop('voxels_dict')
        # TODO: Split predict semantic and bbox
        results_list = self.predict_bbox(feats_dict, voxels_dict,
                                         batch_input_metas, rpn_results_list,
                                         self.test_cfg)
        return results_list

    def predict_bbox(self, feats_dict: Dict, voxel_dict: Dict,
                     batch_input_metas: List[dict],
                     rpn_results_list: InstanceList,
                     test_cfg: ConfigDict) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxel_dict (dict): Contains information of voxels.
            batch_input_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            test_cfg (Config): Test config.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        ...

    def loss(self, feats_dict: Dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        losses = dict()
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        voxels_dict = feats_dict.pop('voxels_dict')
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict, voxels_dict, batch_gt_instances_3d)
            losses.update(semantic_results.pop('loss_semantic'))

        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d)
        if self.with_bbox:
            feats_dict.update(semantic_results)
            bbox_results = self._bbox_forward_train(feats_dict, voxels_dict,
                                                    sample_results)
            losses.update(bbox_results['loss_bbox'])

        return losses
```

姝ゅ鎴戜滑鐪佺暐浜嗙浉鍏冲嚱鏁扮殑鏇村缁嗚妭銆傛洿澶氱粏鑺傝鍙傝€僛浠ｇ爜](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/models/roi_heads/part_aggregation_roi_head.py)銆?
鏈€鍚庯紝鐢ㄦ埛闇€瑕佸湪 `mmdet3d/models/roi_heads/bbox_heads/__init__.py` 鍜?`mmdet3d/models/roi_heads/__init__.py` 娣诲姞妯″潡锛屼粠鑰岃兘琚浉搴旂殑娉ㄥ唽鍣ㄦ壘鍒板苟鍔犺浇銆?
姝ゅ锛岀敤鎴蜂篃鍙互鍦ㄩ厤缃枃浠朵腑娣诲姞浠ヤ笅浠ｇ爜浠ヨ揪鍒扮浉鍚岀殑鐩殑銆?
```python
custom_imports=dict(
    imports=['mmdet3d.models.roi_heads.part_aggregation_roi_head', 'mmdet3d.models.roi_heads.bbox_heads.parta2_bbox_head'],
    allow_failed_imports=False)
```

`PartAggregationROIHead` 鐨勯厤缃枃浠跺涓嬫墍绀猴細

```python
model = dict(
    ...
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=3,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_part=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='avg')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[256, 512, 512, 512],
            cls_channels=[256, 256],
            reg_channels=[256, 256],
            dropout_ratio=0.1,
            roi_feat_size=14,
            with_corner_loss=True,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    ...
)
```

MMDetection 2.0 寮€濮嬫敮鎸侀厤缃枃浠朵箣闂寸殑缁ф壙锛屽洜姝ょ敤鎴峰彲浠ュ叧娉ㄩ厤缃枃浠剁殑淇敼銆侾artA2 Head 鐨勭浜岄樁娈典富瑕佷娇鐢ㄤ簡鏂扮殑 `PartAggregationROIHead` 鍜?`PartA2BboxHead`锛岄渶瑕佹牴鎹搴旀ā鍧楃殑 `__init__` 鍑芥暟鏉ヨ缃弬鏁般€?
### 娣诲姞鏂扮殑鎹熷け鍑芥暟

鍋囪鎮ㄦ兂瑕佷负妫€娴嬫鐨勫洖褰掓坊鍔犱竴涓柊鐨勬崯澶卞嚱鏁?`MyLoss`銆備负浜嗘坊鍔犱竴涓柊鐨勬崯澶卞嚱鏁帮紝鐢ㄦ埛闇€瑕佸湪 `mmdet3d/models/losses/my_loss.py` 涓疄鐜拌鍑芥暟銆傝楗板櫒 `weighted_loss` 鑳藉淇濊瘉瀵规瘡涓厓绱犵殑鎹熷け杩涜鍔犳潈骞冲潎銆?
```python
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from mmdet3d.registry import MODELS


@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@MODELS.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

鎺ヤ笅鏉ワ紝鐢ㄦ埛闇€瑕佸湪 `mmdet3d/models/losses/__init__.py` 娣诲姞璇ュ嚱鏁般€?
```python
from .my_loss import MyLoss, my_loss
```

鎴栬€呭湪閰嶇疆鏂囦欢涓坊鍔犱互涓嬩唬鐮佷互杈惧埌鐩稿悓鐨勭洰鐨勩€?
```python
custom_imports=dict(
    imports=['mmdet3d.models.losses.my_loss'],
    allow_failed_imports=False)
```

涓轰簡浣跨敤璇ュ嚱鏁帮紝鐢ㄦ埛闇€瑕佷慨鏀?`loss_xxx` 鍩熴€傜敱浜?`MyLoss` 鏄敤浜庡洖褰掔殑锛屾偍闇€瑕佷慨鏀?head 涓殑 `loss_bbox` 鍩熴€?
```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0)
```

