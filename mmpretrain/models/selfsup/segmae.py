# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union,Dict
# import math
# import cv2
import torch
from mmengine.structures import BaseDataElement
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor

# from mmcls.models import VisionTransformer
# import numpy as np
# from ..utils import build_2d_sincos_position_embedding
#
# import torchvision.transforms as transform
@MODELS.register_module()
class SegMAE(BaseSelfSupervisor):
    """SegMAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self,
                     inputs: List[torch.Tensor],
                     data_samples: Optional[List[DataSample]] = None,
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """

        latent, mask, ids_restore = self.backbone(inputs, data_samples)
        pred = self.neck(latent, ids_restore)
        self.mask = mask
        return pred



    def reconstruct(self,
                    features: torch.Tensor,
                    data_samples: Optional[List[DataSample]] = None,
                    **kwargs) -> DataSample:
        """The function is for image reconstruction.

        Args:
            features (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            DataSample: The prediction from model.
        """
        mean = kwargs['mean']
        std = kwargs['std']
        features = features * std + mean

        pred = self.head.unpatchify(features)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = self.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        results = DataSample()
        results.mask = BaseDataElement(**dict(value=mask))
        results.pred = BaseDataElement(**dict(value=pred))

        return results

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        # print('inputs',inputs)
        # print('data_samples',data_samples)
        latent, mask, ids_restore = self.backbone(inputs, data_samples)
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, inputs, mask)
        losses = dict(loss=loss)
        return losses
