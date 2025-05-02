import os
import xml.etree.ElementTree as ET
import argparse

def find_images_with_object(voc_root, object_name):
    """
    Finds image filenames containing a specific object in the PASCAL VOC dataset.

    Args:
        voc_root (str): The root directory of the PASCAL VOC dataset (e.g., /path/to/VOCdevkit/VOC2012).
        object_name (str): The name of the object to search for (e.g., 'sofa').

    Returns:
        list: A list of image filenames (without extension) containing the object.
    """
    annotations_dir = os.path.join(voc_root, 'Annotations')
    image_filenames = []

    if not os.path.isdir(annotations_dir):
        print(f"Error: Annotations directory not found at {annotations_dir}")
        return []

    print(f"Searching for '{object_name}' in annotations at: {annotations_dir}")

    # 모든 XML 주석 파일 순회
    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.xml'):
            continue

        xml_path = os.path.join(annotations_dir, filename)

        try:
            # XML 파일 파싱
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # XML 내의 모든 'object' 태그 확인
            found = False
            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if name_tag is not None and name_tag.text == object_name:
                    found = True
                    break # 해당 객체를 찾으면 더 이상 이 파일 내에서 찾을 필요 없음

            # 객체를 찾았다면 이미지 파일 이름 저장 (확장자 제외)
            if found:
                image_filename = os.path.splitext(filename)[0]
                image_filenames.append(image_filename)

        except ET.ParseError:
            print(f"Warning: Could not parse XML file: {xml_path}")
        except Exception as e:
            print(f"An error occurred processing {xml_path}: {e}")

    return image_filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find images containing a specific object in PASCAL VOC 2012.")
    # --- 수정: VOC 데이터셋 루트 경로를 인자로 받도록 변경 ---
    parser.add_argument('voc_root', help='Path to the VOCdevkit/VOC2012 directory')
    parser.add_argument('--object', default='sofa', help="Object name to search for (default: 'sofa')")
    args = parser.parse_args()

    # --- 수정: 인자로 받은 경로 사용 ---
    # voc_dataset_path = '/nas_homes/dataset/VOC2012' # 하드코딩 대신 인자 사용
    voc_dataset_path = args.voc_root
    object_to_find = args.object

    sofa_images = find_images_with_object(voc_dataset_path, object_to_find)

    if sofa_images:
        print(f"\nFound {len(sofa_images)} images containing '{object_to_find}':")
        for img_name in sofa_images:
            # 전체 이미지 경로 출력 (선택 사항)
            img_path = os.path.join(voc_dataset_path, 'JPEGImages', f"{img_name}.jpg")
            print(f"- Filename: {img_name}, Path: {img_path}")
            # print(img_name) # 파일 이름만 출력하려면
    else:
        print(f"\nNo images containing '{object_to_find}' found in the annotations.")
