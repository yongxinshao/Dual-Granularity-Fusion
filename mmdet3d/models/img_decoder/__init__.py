# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .deepv3plus import DepthwiseSeparableASPPHead
from .decoder_head import BaseDecodeHead
__all__ = [
    'DepthwiseSeparableASPPHead', 'ASPPHead','BaseDecodeHead'
]

