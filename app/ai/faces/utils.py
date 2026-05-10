from typing import List, Dict


def face_box_area(face: Dict) -> int:
    x1, y1, x2, y2 = face["box"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_faces_by_area(faces: List[Dict], min_area: int) -> List[Dict]:
    """
    min_area 픽셀^2 이상의 얼굴만 남긴다.
    예: min_area = 80 * 80  -> 80x80 이상 얼굴만
    """
    return [f for f in faces if face_box_area(f) >= min_area]
