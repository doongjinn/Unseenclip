# configs/_base_/models/clip_rc.py

# --- 이 부분은 데이터셋에 따라 달라집니다 ---
# 예시: Pascal VOC 데이터셋 (20개 클래스 + 배경)
# seen_idx = [1, 2, ..., 15]  # 예시 seen 클래스 인덱스
# all_idx = [0, 1, ..., 20]   # 예시 모든 클래스 인덱스 (배경 포함)
# contrastive_weight = 0.5   # 예시 가중치

# --- 실제 값으로 채워야 합니다 ---
_seen_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # 데이터셋의 seen 클래스 인덱스 리스트
_all_idx = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
]  # 데이터셋의 모든 클래스 인덱스 리스트 (배경 포함)
_contrastive_weight = 1.0 # 필요에 따라 조정

img_size = 512
in_channels = 768 # 백본 width와 일치하도록 수정 (이전 논의 기반)
out_indices = [11]

model = dict(type='CLIPRC',
             pretrained='ViT-B-16.pt',
             backbone=dict(type='CLIPVisionTransformerWithRLB',
                           layers=12,
                           width=768, # 백본 width 명시
                           get_embeddings=False, # 차원 일치를 위해 False로 설정 (이전 논의 기반)
                           style='pytorch'),
             text_encoder=dict(type='CLIPTextEncoder',
                               embed_dim=768, # 텍스트 인코더 출력 차원 명시 (백본/디코더와 일치)
                               context_length=77,
                               style='pytorch'),
             decode_head=dict(
                 type='ATMSingleHeadSeg',
                 img_size=img_size,
                 in_channels=in_channels, # 백본 width와 일치
                 embed_dims=in_channels,  # 디코더 내부 차원 (백본 width와 일치)
                 num_layers=3,
                 num_heads=8,
                 use_stages=len(out_indices),
                 loss_decode=dict(type='SegLoss',
                                  dec_layers=3,
                                  loss_weight=1.0,
                                  seen_idx=_seen_idx,       # 추가
                                  all_idx=_all_idx,         # 추가
                                  contrastive_weight=_contrastive_weight # 추가
                                  ),
             ),
             train_cfg=dict(),
             test_cfg=dict(mode='slide',
                           crop_size=(512, 512),
                           stride=(426, 426)))
