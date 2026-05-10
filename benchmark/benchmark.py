"""
AI 모델 자동 벤치마크
모든 조합을 테스트하고 FPS 비교
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.detectors import get_detector
from app.ai.trackers import get_tracker
from app.ai.inference import AIInference


class BenchmarkRunner:
    """
    벤치마크 실행기
    """

    def __init__(
        self, num_frames: Optional[int] = None, image_folder: Optional[str] = None
    ):
        self.num_frames = num_frames
        self.image_folder = image_folder

        if self.image_folder is None and self.num_frames is None:
            self.num_frames = 100  # 기본값 설정
        elif self.image_folder is not None:
            if self.num_frames is not None:
                print(f"⚠️ image folder detected, ignoring num_frames")

            # 이미지 로딩
            self.images = self.load_images(self.image_folder)
            if not self.images:
                raise RuntimeError(
                    f"No valid images found in folder: {self.image_folder}"
                )
            self.num_frames = len(self.images)

        self.results = []

    def load_images(self, folder_path: str) -> List[np.ndarray]:
        """
        폴더에서 이미지들을 읽어 리스트로 반환 (허용 확장자만).
        이미지 읽기 실패 시 건너뜀.
        """
        p = Path(folder_path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        images: List[np.ndarray] = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  ⚠️ Skipping unreadable image: {f.name}")
                continue
            images.append(img)

        return images

    def generate_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        테스트용 더미 이미지 생성
        """
        # 랜덤 이미지 (실제 카메라 프레임 시뮬레이션)
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return image

    def benchmark_combination(self, detector_name: str, tracker_name: str) -> Dict:
        """
        특정 조합 벤치마크

        Returns:
            {
                "detector": "yolov8n",
                "tracker": "bytetrack",
                "avg_fps": 25.3,
                "min_fps": 18.2,
                "max_fps": 32.1,
                "total_time": 3.95
            }
        """
        print(f"\n{'=' * 60}")
        print(f"Testing: {detector_name} + {tracker_name}")
        print(f"{'=' * 60}")

        try:
            # AI 인스턴스 생성
            ai = AIInference(detector_name, tracker_name)

            # 워밍업
            dummy = self.generate_test_image()
            ai.process_image(dummy)

            # 벤치마크
            fps_list = []
            start_total = time.time()

            for i in range(self.num_frames):
                if self.image_folder is not None:
                    # 폴더의 모든 이미지를 순환해서 사용
                    image = self.images[i]
                else:
                    image = self.generate_test_image()

                start = time.time()
                tracks = ai.process_image(image)
                elapsed = time.time() - start

                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)

                if (i + 1) % 10 == 0:
                    avg = sum(fps_list[-10:]) / 10
                    print(
                        f"  Frame {i + 1}/{self.num_frames}: {fps:.2f} FPS (avg: {avg:.2f})"
                    )

            total_time = time.time() - start_total

            result = {
                "detector": detector_name,
                "tracker": tracker_name,
                "avg_fps": sum(fps_list) / len(fps_list),
                "min_fps": min(fps_list),
                "max_fps": max(fps_list),
                "total_time": total_time,
                "frames": self.num_frames,
            }

            print(f"\n✅ Result:")
            print(f"  Average FPS: {result['avg_fps']:.2f}")
            print(f"  Min FPS: {result['min_fps']:.2f}")
            print(f"  Max FPS: {result['max_fps']:.2f}")
            print(f"  Total Time: {result['total_time']:.2f}s")

            return result

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return {"detector": detector_name, "tracker": tracker_name, "error": str(e)}

    def run_all_combinations(self):
        """
        모든 조합 테스트
        """
        detectors = ["yolov8n", "yolov11n", "yolov5n", "rtdetr", "effdetlite"] 
        trackers = ["bytetrack", "deepsort"]  

        print(f"\n🚀 Starting Benchmark")
        print(f"Frames per test: {self.num_frames}")
        print(f"Combinations: {len(detectors)} detectors × {len(trackers)} trackers")

        for detector in detectors:
            for tracker in trackers:
                result = self.benchmark_combination(detector, tracker)
                self.results.append(result)

        self.print_summary()

    def print_summary(self):
        """
        결과 요약 출력
        """
        print(f"\n\n{'=' * 80}")
        print(f"📊 BENCHMARK SUMMARY")
        print(f"{'=' * 80}\n")

        # 표 형식 출력
        print(
            f"{'Detector':<15} {'Tracker':<15} {'Avg FPS':>10} {'Min FPS':>10} {'Max FPS':>10}"
        )
        print(f"{'-' * 70}")

        # FPS 기준 정렬
        sorted_results = sorted(
            [r for r in self.results if "error" not in r],
            key=lambda x: x["avg_fps"],
            reverse=True,
        )

        for i, result in enumerate(sorted_results, 1):
            print(
                f"{result['detector']:<15} {result['tracker']:<15} "
                f"{result['avg_fps']:>10.2f} {result['min_fps']:>10.2f} {result['max_fps']:>10.2f}"
            )

            if i == 1:
                print(f"  ⭐ FASTEST!")

        # 에러 결과
        error_results = [r for r in self.results if "error" in r]
        if error_results:
            print(f"\n❌ Failed combinations:")
            for result in error_results:
                print(
                    f"  {result['detector']} + {result['tracker']}: {result['error']}"
                )

        print(f"\n{'=' * 80}\n")


def main():
    """
    메인 함수
    """
    import argparse

    parser = argparse.ArgumentParser(description="AI Model Benchmark")
    parser.add_argument("--frames", type=int, help="Number of frames to test")
    parser.add_argument("--detector", type=str, help="Test specific detector")
    parser.add_argument("--tracker", type=str, help="Test specific tracker")
    parser.add_argument(
        "--image_folder", type=str, help="Path to folder with images for benchmarking"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(num_frames=args.frames, image_folder=args.image_folder)

    if args.detector and args.tracker:
        # 특정 조합만 테스트
        runner.benchmark_combination(args.detector, args.tracker)
    else:
        # 모든 조합 테스트
        runner.run_all_combinations()


if __name__ == "__main__":
    main()
