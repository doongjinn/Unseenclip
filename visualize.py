# visualize_single_with_original.py
import argparse
import os
import os.path as osp # 경로 처리를 위해 추가
import mmcv
import torch # For progress bar
from mmcv.utils import Config # Config 클래스 import 추가
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel # For single GPU inference
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_palette # 체크포인트에 팔레트 없을 경우 대비
from mmseg.datasets import build_dataset, build_dataloader

# 사용자 정의 모델/모듈이 있다면 import 필요
import models # models/__init__.py 가 모듈을 등록하는지 확인

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model and save visualizations')
    parser.add_argument('config', help='Model config file path used for training')
    parser.add_argument('checkpoint', help='Trained checkpoint file path (e.g., latest.pth)')
    parser.add_argument('output_dir', help='Directory to save segmentation results and ground truth images') # Changed
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference (e.g., cuda:0 or cpu)')
    parser.add_argument(
        '--palette', default='voc', # 데이터셋에 맞는 팔레트 지정 (voc, cityscapes 등)
        help='Color palette for segmentation map')
    parser.add_argument(
        '--opacity', type=float, default=0.5,
        help='Opacity of the segmentation mask overlay (0-1)')
    args = parser.parse_args()
    return args

def visualize_ground_truth(model_to_show, img_path, gt_path, palette, opacity, out_file):
    """Loads image and ground truth, then visualizes GT using model.show_result and saves it."""
    try:
        img = mmcv.imread(img_path)
        if img is None:
            print(f"Error: Could not read image file: {img_path}")
            return False
        # Load GT segmentation map (usually single-channel png)
        # Use 'pillow' backend as it often handles indexed PNGs better
        gt_seg_map = mmcv.imread(gt_path, flag='unchanged', backend='pillow')
        if gt_seg_map is None:
            print(f"Error: Could not read ground truth file: {gt_path}")
            return False

        # Ensure GT map is 2D (H, W)
        if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 1:
            gt_seg_map = gt_seg_map[:, :, 0]
        elif gt_seg_map.ndim != 2:
             print(f"Error: Unexpected ground truth map dimensions: {gt_seg_map.shape} for {gt_path}")
             return False


        # Ensure palette has enough colors for max index in GT map + 1
        # model.show_result 내부에서 palette 길이를 처리하므로 이 부분은 생략 가능
        # max_idx = gt_seg_map.max()
        # if max_idx >= len(palette):
        #     print(f"Warning: Ground truth map contains index {max_idx}, but palette only has {len(palette)} colors. Clamping indices or using default colors might occur.")

        # Use model.show_result to visualize GT on the image
        # Pass a copy of img if show_result modifies it in-place (usually doesn't)
        # Pass gt_seg_map inside a list/tuple as show_result expects a result iterable
        model_to_show.show_result(
            img.copy(),
            [gt_seg_map], # Pass GT map as a list
            palette=palette,
            opacity=opacity,
            show=False,
            out_file=out_file
        )
        return True
    except Exception as e:
        print(f"Error visualizing ground truth for {osp.basename(img_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parse_args()

    # --- 1. 설정 로드 및 클래스 이름 주입 ---
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)

    # --- 클래스 이름 주입 시도 ---
    class_names = None
    palette = None
    try:
        # Test dataset 우선 사용
        dataset_cfg = cfg.data.get('test', None)
        if dataset_cfg is None:
             print("Warning: 'test' dataset config not found, trying 'val'.")
             dataset_cfg = cfg.data.get('val', None)
        if dataset_cfg is None:
             print("Warning: 'val' dataset config also not found, trying 'train'.")
             dataset_cfg = cfg.data.get('train', None)

        if dataset_cfg is None:
            print("Error: No test, val, or train dataset configuration found in the config file.")
            return

        # 클래스 이름 얻기 (설정 또는 데이터셋 빌드)
        if 'classes' in dataset_cfg:
            class_names = dataset_cfg['classes']
            print(f"Found class names in cfg.data.{dataset_cfg.get('type', 'dataset')}.classes")
        else:
            print("Class names not found directly in config. Building dataset to get classes...")
            # 임시로 dataset_cfg 수정하여 test_mode 설정 (build_dataset 위함)
            # dataset_cfg_copy = dataset_cfg.copy() # 원본 수정 방지
            # dataset_cfg_copy.test_mode = True
            dataset_instance = build_dataset(dataset_cfg) # Use determined config
            if not hasattr(dataset_instance, 'CLASSES'):
                 print("Error: Failed to get CLASSES attribute from built dataset.")
                 return
            class_names = dataset_instance.CLASSES
            print(f"Obtained class names from built dataset: {class_names}")

        if class_names:
            print(f"Injecting class names into model config...")
            cfg.model.class_names = class_names
        else:
            raise ValueError("Could not determine class_names.")

        # 팔레트 얻기 (설정 또는 데이터셋 빌드)
        if 'palette' in dataset_cfg:
             palette = dataset_cfg['palette']
             print(f"Found palette in cfg.data.{dataset_cfg.get('type', 'dataset')}.palette")
        else:
             print(f"Palette not found in config. Trying to get from built dataset or using default '{args.palette}'.")
             if 'dataset_instance' in locals() and hasattr(dataset_instance, 'PALETTE'):
                 palette = dataset_instance.PALETTE
                 print("Using PALETTE from built dataset instance.")
             else:
                 print(f"Generating default palette '{args.palette}' for {len(class_names)} classes.")
                 palette = get_palette(args.palette, len(class_names))

        if not palette:
             raise ValueError("Could not determine PALETTE.")


    except Exception as e:
        print(f"Error determining or injecting class names/palette: {e}")
        print("Please ensure class names and palette are available in the config's data section or can be derived.")
        return

    # --- 2. 모델 초기화 (수정된 cfg 사용) ---
    print(f"Initializing model from modified config...")
    model = init_segmentor(cfg, checkpoint=None, device=args.device)
    model.CLASSES = class_names # Ensure CLASSES are set
    model.PALETTE = palette     # Ensure PALETTE is set

    # --- 3. 체크포인트 로드 ---
    print(f"Loading checkpoint: {args.checkpoint}")
    # map_location을 지정하여 특정 device에 로드 강제 가능
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=args.device)
    # load_checkpoint 후에도 CLASSES/PALETTE 확인 (체크포인트 메타 우선)
    if 'CLASSES' in checkpoint.get('meta', {}):
         model.CLASSES = checkpoint['meta']['CLASSES']
         print("Overrode CLASSES with checkpoint metadata.")
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
        print("Overrode PALETTE with checkpoint metadata.")
    # 최종 확인
    if not hasattr(model, 'CLASSES') or not model.CLASSES or not hasattr(model, 'PALETTE') or not model.PALETTE:
         print("Error: Model CLASSES or PALETTE is still missing after setup.")
         return

    model.cfg = cfg # cfg 속성 설정 (show_result 등에서 사용될 수 있음)
    model.eval()

    # Wrap the model for single GPU inference (consistent with multi-GPU training)
    # Convert device ID string to integer
    device_id = int(args.device.split(':')[1])
    model = MMDataParallel(model, device_ids=[device_id])


    # --- 4. 테스트 데이터셋 및 데이터로더 준비 ---
    print("Building test dataset...")
    # Use the dataset_cfg determined earlier
    test_dataset = build_dataset(dataset_cfg)
    print(f"Building test dataloader (Batch size: 1)...")
    # samples_per_gpu=1, workers_per_gpu=1 로 단일 이미지 처리 설정
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=1, # 병렬 로딩 워커 수 (필요에 따라 조정)
        dist=False,        # 분산 처리 안 함
        shuffle=False)     # 순서대로 처리

    # --- 5. 출력 디렉토리 생성 ---
    mmcv.mkdir_or_exist(osp.abspath(args.output_dir))
    print(f"Output directory: {osp.abspath(args.output_dir)}")

    # --- 6. 데이터셋 순회 및 추론/시각화 ---
    model_to_show = model.module # MMDataParallel 로 래핑된 모델 내부 접근
    num_processed = 0
    num_skipped = 0

    # Progress bar setup
    prog_bar = mmcv.ProgressBar(len(test_dataset))

    for i, data in enumerate(test_dataloader):
        try:
            # data['img_metas']는 DataContainer 객체를 담은 리스트
            img_metas_dc = data['img_metas'][0] # 리스트의 첫 요소인 DataContainer 객체 가져오기
            # DataContainer 내부의 실제 데이터(아마도 [[딕셔너리]] 형태) 가져오기
            img_metas_nested_list = img_metas_dc.data
            # 실제 메타데이터 딕셔너리 가져오기 (이중 리스트 접근)
            img_metas = img_metas_nested_list[0][0] # 이중 리스트 안의 딕셔너리 접근
            # 이제 딕셔너리에서 키로 접근 가능
            img_path = img_metas['filename']
            gt_path = img_metas.get('ann_file', None) # Get GT path if available

            # <<<--- 수정 시작: gt_path가 없을 경우 추측 시도 --->>>
            if gt_path is None:
                print(f"Warning: 'ann_file' not found in metadata for {osp.basename(img_path)}. Attempting to guess GT path based on standard VOC structure.")
                try:
                    # VOC 표준 구조 가정: /path/to/VOC2012/JPEGImages/xxxxx.jpg -> /path/to/VOC2012/SegmentationClass/xxxxx.png
                    img_dir_part = 'JPEGImages'
                    gt_dir_part = 'SegmentationClass'
                    img_ext = '.jpg'
                    gt_ext = '.png'

                    if img_dir_part in img_path and img_path.endswith(img_ext):
                        potential_gt_path = img_path.replace(img_dir_part, gt_dir_part).replace(img_ext, gt_ext)
                        if osp.exists(potential_gt_path):
                            gt_path = potential_gt_path # 추측 성공 및 파일 존재
                            print(f"Successfully guessed GT path: {gt_path}")
                        else:
                            # 추측은 했으나 해당 경로에 파일 없음
                            print(f"Guessed GT path ({potential_gt_path}) does not exist.")
                            gt_path = None # 실패 처리
                    else:
                        # 이미지 경로가 예상 패턴과 다름 (예: 확장자가 다르거나 JPEGImages 폴더가 아님)
                        print(f"Could not apply standard path guessing logic to: {img_path}")
                        gt_path = None # 실패 처리
                except Exception as e_guess:
                    # 경로 추측 중 예외 발생
                    print(f"Error occurred while trying to guess GT path: {e_guess}")
                    gt_path = None # 실패 처리
            # <<<--- 수정 끝 --->>>

            if not osp.exists(img_path):
                 print(f"Warning: Image file not found: {img_path}. Skipping.")
                 num_skipped += 1
                 prog_bar.update()
                 continue

            # --- 6.1. 추론 수행 ---
            # MMDataParallel 래퍼 내부의 실제 모델(.module)을 전달
            result = inference_segmentor(model.module, img_path)

            # --- 6.2. 결과 시각화 및 저장 ---
            base_name = osp.splitext(osp.basename(img_path))[0]
            pred_save_path = osp.join(args.output_dir, f"{base_name}_pred.png")

            img = mmcv.imread(img_path) # 원본 이미지 로드 (show_result 용)
            if img is None:
                 print(f"Warning: Failed to read image {img_path} for visualization. Skipping prediction save.")
                 num_skipped += 1
                 prog_bar.update()
                 continue

            # show_result 사용 (model_to_show는 내부 모델)
            model_to_show.show_result(
                img,
                result,
                palette=model.module.PALETTE, # Use the verified palette from the underlying model
                show=False,
                out_file=pred_save_path,
                opacity=args.opacity
            )

            # --- 6.3. Ground Truth 시각화 및 저장 ---
            # 이제 gt_path는 메타데이터에서 왔거나, 추측되었거나, 여전히 None일 수 있음
            if gt_path and osp.exists(gt_path): # 경로가 있고 파일도 존재하면 시각화 시도
                gt_save_path = osp.join(args.output_dir, f"{base_name}_gt.png")
                # visualize_ground_truth 호출 시 model_to_show 전달
                gt_saved = visualize_ground_truth(
                    model_to_show, # Pass the model
                    img_path,
                    gt_path, # 메타데이터 또는 추측된 경로 사용
                    model.module.PALETTE, # Use the same palette from the underlying model
                    args.opacity,
                    gt_save_path
                )
                if not gt_saved:
                    print(f"Warning: Failed to visualize ground truth for {osp.basename(img_path)} using path {gt_path}")
                    # GT 저장 실패해도 예측 저장은 했으므로 계속 진행
            elif gt_path: # 경로는 있었으나(추측 포함) 파일이 없거나 처리 실패
                print(f"Warning: Ground truth file path found/guessed ({gt_path}) but the file doesn't exist or couldn't be processed. Cannot save GT visualization.")
            else: # 메타데이터에도 없고 추측도 실패함
                 print(f"Warning: No ground truth path ('ann_file') found in metadata and guessing failed for {osp.basename(img_path)}. Cannot save GT visualization.")


            num_processed += 1
        except Exception as e:
            print(f"\nError processing item {i} ({osp.basename(img_path if 'img_path' in locals() else 'unknown')}): {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            num_skipped += 1
            # 오류 발생 시 다음 이미지로 계속 진행할 수 있도록 함

        # Update progress bar
        prog_bar.update()

    print(f"\nProcessing finished.")
    print(f"Successfully processed and saved results for {num_processed} images.")
    if num_skipped > 0:
        print(f"Skipped {num_skipped} images due to errors.")
    print(f"Outputs saved in: {osp.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()