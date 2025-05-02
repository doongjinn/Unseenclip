import torch
import torch.nn.functional as F
import numpy as np
import os # Import os for path joining

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from .utils import tokenize


@SEGMENTORS.register_module()
class CLIPRC(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 text_encoder,
                 pretrained_text,
                 class_names,
                 base_class,
                 novel_class,
                 both_class,
                 multi_prompts=False,
                 self_training=False,
                 ft_backbone=False,
                 exclude_key=None,
                 load_text_embedding=None,
                 clip_cls_features_path=None,
                 dino_features_path=None,
                 **args):
        super(CLIPRC, self).__init__(**args)

        if pretrained_text is not None:
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained_text

        self.text_encoder = builder.build_backbone(text_encoder)
        
        # Load pre-extracted DINOv2 features only for training
        self.dinov2_features = None
        if dino_features_path is not None and self.training:
            # Construct the full path relative to the workspace root if necessary
            # Assuming dinov2_features_path might be relative
            # full_dinov2_path = os.path.join(os.getcwd(), dinov2_features_path) 
            # It's usually better if the config provides the absolute path or a path relative to a known root
            if os.path.exists(dino_features_path):
                print(f"Loading DINOv2 features from: {dino_features_path}")
                self.dinov2_features = torch.load(dino_features_path, map_location='cpu') # Load to CPU initially
            else:
                print(f"Warning: DINOv2 features path not found: {dino_features_path}")
                self.dinov2_features = None # Ensure it's None if file not found

        self.class_names = class_names
        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.multi_prompts = multi_prompts
        self.load_text_embedding = load_text_embedding

        if not self.load_text_embedding:
            if not self.multi_prompts:
                self.texts = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in self.class_names])
            else:
                self.texts = self._get_multi_prompts(self.class_names)

        if len(self.base_class) != len(self.both_class):  # zero-shot setting
            if not self_training:
                self._visiable_mask(self.base_class)
            else:
                self._visiable_mask_st(self.base_class)
                self._st_mask(self.novel_class)

        if self.training:
            self._freeze_stages(self.text_encoder)
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
            print('--------------------------------------')
            for n, m in self.named_parameters():
                if m.requires_grad:
                    print('Finetune layer in segmentor:', n)
        else:
            self.text_encoder.eval()
            self.backbone.eval()

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError(
                        "Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting:',
              self.visibility_seen_mask)

    def _visiable_mask_st(self, seen_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = n
        seen_map[200] = 200  # pixels of padding will be excluded
        self.visibility_seen_mask = seen_map.copy()
        print(
            'Making visible mask for zero-shot setting in self_traning stage:',
            self.visibility_seen_mask)

    def _st_mask(self, novel_classes):
        st_mask = np.array([255] * 256)
        st_mask[255] = 255
        for i, n in enumerate(list(novel_classes)):
            st_mask[n] = n
        self.st_mask = st_mask.copy()
        print('Making st mask for zero-shot setting in self_traning stage:',
              self.st_mask)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _decode_head_forward_train(self, feat, img_metas, gt_semantic_seg, train_cfg, 
                                   unseen_token=None, dino_cls_token=None):
        """Run forward function and calculate loss for decode head in
        training."""
        if self.training:
            if len(self.base_class) != len(self.both_class):  # zero setting
                gt_semantic_seg = torch.Tensor(
                    self.visibility_seen_mask).type_as(
                        gt_semantic_seg)[gt_semantic_seg]

        losses = dict()
        if self.self_training:
            loss_decode = self.decode_head.forward_train(
                feat, img_metas, gt_semantic_seg, train_cfg,
                self.self_training,
                st_mask=self.st_mask
            )
        else:
            loss_decode = self.decode_head.forward_train(
                feat, img_metas, gt_semantic_seg, train_cfg,
                self.self_training,
                unseen_token=unseen_token,
                dino_cls_token=dino_cls_token
                )

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def text_embedding(self, texts, img):
        text_embeddings = self.text_encoder(texts.to(img.device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1,
                                                                 keepdim=True)
        return text_embeddings

    def extract_feat(self, img):
        """Extract features from images."""
        features = self.backbone(img)
        return {
            'visual_features': features[0],
            'cls_token': features[1],
            'unseen_token': features[2]
        }

    def _get_dino_features(self, img_metas, device):
        """Get DINOv2 features for the current batch of images."""
        if self.dinov2_features is None or not self.training:
            return None
            
        batch_features = []
        not_found_count = 0
        for meta in img_metas:
            filename_key = meta.get('ori_filename', meta.get('filename'))
            if filename_key is None:
                print("Warning: Could not determine filename from img_meta. Cannot retrieve DINOv2 features.")
                return None
            try:
                img_key = os.path.splitext(os.path.basename(filename_key))[0]
            except Exception as e:
                print(f"Warning: Error extracting img_key from {filename_key}. Error: {e}")
                img_key = None

            if img_key and img_key in self.dinov2_features:
                batch_features.append(self.dinov2_features[img_key].to(device))
            else:
                not_found_count += 1
                try:
                    example_feat = next(iter(self.dinov2_features.values()))
                    placeholder = torch.zeros_like(example_feat, device=device)
                except StopIteration:
                    print("Warning: DINOv2 feature dictionary is empty. Cannot create placeholder.")
                    placeholder_dim = getattr(self.backbone, 'output_dim', 768)
                    placeholder = torch.zeros(placeholder_dim, device=device)
                batch_features.append(placeholder)

        if not_found_count > 0 and rank == 0:
            print(f"Warning: DINOv2 features not found or failed to extract key for {not_found_count}/{len(img_metas)} images in the batch.")
            
        if not batch_features:
             print("Warning: No DINOv2 features could be collected for the batch.")
             return None
        try:
            return torch.stack(batch_features)
        except RuntimeError as e:
             print(f"Error stacking DINOv2 features: {e}. Check feature dimensions.")
             for i, feat in enumerate(batch_features):
                 print(f"Feature {i} shape: {feat.shape}")
             return None

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training."""
        features = self.extract_feat(img)
        
        # Get text features
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device)
        else:
            text_feat = self.text_embedding(self.texts, img)

        # Format features as expected by decoder
        feat = []
        feat.append([
            features['visual_features'],
            features['cls_token'],
            features['unseen_token']
        ])
        feat.append(text_feat)
        
        # Get DINOv2 features for the batch
        dino_cls_token = self._get_dino_features(img_metas, img.device)

        # --- ADDED: Detach dino_cls_token to prevent grads flowing back ---
        if dino_cls_token is not None:
            dino_cls_token = dino_cls_token.detach()
        # ------------------------------------------------------------------

        losses = self._decode_head_forward_train(
            feat, 
            img_metas,
            gt_semantic_seg,
            self.train_cfg,
            unseen_token=features['unseen_token'],
            dino_cls_token=dino_cls_token
            )
            
        return losses

    def encode_decode(self, img, img_metas):
        # Extract visual features
        features = self.extract_feat(img)

        # Get text features (similar to forward_train)
        if self.load_text_embedding:
            # Ensure text_feat is loaded or cached appropriately during inference
            # For simplicity, assuming it might be pre-loaded if needed
            # Or recalculate if necessary (check performance implications)
            if not hasattr(self, '_cached_text_feat') or self._cached_text_feat is None:
                 try: 
                     text_feat = np.load(self.load_text_embedding)
                     self._cached_text_feat = torch.from_numpy(text_feat).to(img.device)
                 except FileNotFoundError:
                      print(f"Warning: Text embedding file {self.load_text_embedding} not found during encode_decode. Attempting to generate.")
                      if hasattr(self, 'texts'):
                          self._cached_text_feat = self.text_embedding(self.texts, img)
                      else: # Fallback if texts not available
                           print("Error: Cannot get text features during inference.")
                           # Return zeros or raise error, depending on desired behavior
                           # Returning zeros based on the shape of visual features
                           num_classes = self.num_classes # Get num_classes from __init__
                           embed_dim = features['cls_token'].shape[-1] # Get embed_dim
                           self._cached_text_feat = torch.zeros((num_classes, embed_dim), device=img.device)
            text_feat = self._cached_text_feat
        else:
            # Ensure self.texts is initialized
            if not hasattr(self, 'texts'):
                 self.texts = torch.cat([tokenize(f"a photo of a {c}") for c in self.class_names])
            text_feat = self.text_embedding(self.texts, img)

        # Format features for the decoder head (similar to forward_train)
        # Note: DINO features are typically not used during inference/testing
        feat = []
        feat.append([
            features['visual_features'],
            features['cls_token'],
            features['unseen_token'] 
        ])
        feat.append(text_feat)

        # Call the decoder head's forward_test method
        # Pass the formatted features list/tuple as the primary input 'x'
        # Assuming self_training is False during standard inference
        seg_logits = self._decode_head_forward_test(feat, img_metas, self_training=False)
        return seg_logits

    def _decode_head_forward_test(self, x, img_metas, self_training):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg,
                                                   self_training)
        return seg_logits

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(preds,
                           size=img_meta[0]['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds
