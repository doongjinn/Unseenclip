import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from .criterion import SegPlusCriterion


@LOSSES.register_module()
class SegLoss(nn.Module):
    '''
    modified from https://github.com/ZiqinZhou66/ZegCLIP/blob/main/models/losses/atm_loss.py 
    '''

    def __init__(self,
                 dec_layers,
                 seen_idx,
                 all_idx,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 contrastive_weight=1.0,
                 loss_weight=1.0,
                 dino_embed_dim=768,
                 clip_output_dim=512):
        super(SegLoss, self).__init__()
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({
                k + f"_{i}": v
                for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

        self.criterion = SegPlusCriterion(
            weight_dict=weight_dict,
            losses=["masks"],
        )
        self.loss_weight = loss_weight
        self.contrastive_weight = contrastive_weight

        self.all_idx = all_idx
        self.seen_idx = seen_idx
        unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            if i_idx in unseen_idx:
                unseen_idx.remove(i_idx)
        self.unseen_idx = set(unseen_idx)

        self.contrastive_loss_func = nn.CosineEmbeddingLoss(reduction='mean')

        self.clip_output_dim = clip_output_dim
        self.dino_proj = nn.Linear(dino_embed_dim, clip_output_dim)

    def forward(
        self,
        outputs,
        label,
        ignore_index=255,
        unseen_token=None,
        dino_cls_token=None,
    ):
        """Forward function."""

        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[
                    k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        if unseen_token is not None and dino_cls_token is not None:
            # Ensure tensors are float32 for cosine embedding loss and projection
            unseen_token = unseen_token.float()
            dino_cls_token = dino_cls_token.float()
            
            # Ensure projection layer is on the same device as the tokens
            self.dino_proj.to(dino_cls_token.device)
            projected_dino_token = self.dino_proj(dino_cls_token)
            
            B = label.shape[0]
            has_unseen_list = []
            for i in range(B):
                unique_labels_img = torch.unique(label[i])
                is_present = any(l.item() in self.unseen_idx for l in unique_labels_img if l.item() != self.ignore_index)
                has_unseen_list.append(is_present)
            has_unseen_mask = torch.tensor(has_unseen_list, device=label.device, dtype=torch.bool)
            
            target = torch.where(has_unseen_mask, 
                                 torch.tensor(1.0, device=label.device), 
                                 torch.tensor(-1.0, device=label.device))
            
            contrastive_loss = self.contrastive_loss_func(unseen_token, projected_dino_token, target)
            
            losses['loss_contrastive_cosine'] = contrastive_loss * self.contrastive_weight

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append({
                "labels": gt_cls,
                "target_masks": masks,
                "masks": targets_per_image,
            })
        return new_targets
