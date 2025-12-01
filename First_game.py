"""
=============================================================================
얼굴 3D 스캐너 프로그램
=============================================================================

이 프로그램은 맥북 카메라로 얼굴을 여러 각도에서 찍어서
3D 모델을 만들어주는 프로그램입니다.

만든 날짜: 2024년
대상: 코딩 초보자
환경: 맥북 Pro M3 / Air M1, macOS Tahoe 26

=============================================================================
"""

# =============================================================================
# 1부: 필요한 도구들 불러오기 (import)
# =============================================================================
# 
# import는 "다른 사람이 만들어 놓은 도구를 가져와서 쓸게요"라는 뜻이에요.
# 마치 요리할 때 칼, 도마, 프라이팬을 가져오는 것처럼요.

import cv2                  # OpenCV: 카메라와 이미지 처리 도구
import numpy as np          # NumPy: 숫자 계산 도구
import os                   # OS: 폴더 만들기, 파일 관리 도구
import time                 # Time: 시간 관련 도구 (카운트다운에 사용)
from datetime import datetime  # DateTime: 현재 날짜/시간 알아내는 도구

# =============================================================================
# 2부: 전역 설정값들
# =============================================================================
#
# 프로그램 전체에서 사용할 설정값들을 미리 정해놓는 곳이에요.
# 나중에 이 숫자들만 바꾸면 프로그램 동작을 쉽게 조절할 수 있어요.

# 사진 저장할 폴더 이름
PHOTO_FOLDER = "captured_faces"

# 찍을 사진 개수 (최소/최대)
MIN_PHOTOS = 10
MAX_PHOTOS = 20

# 카메라 해상도 설정
CAMERA_WIDTH = 1280   # 가로 픽셀
CAMERA_HEIGHT = 720   # 세로 픽셀

# 얼굴 인식용 설정
# Haar Cascade는 OpenCV가 제공하는 얼굴 인식 알고리즘이에요
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# =============================================================================
# 3부: 클래스 정의 - Face3DScanner
# =============================================================================
#
# 클래스(class)란?
# - 관련된 기능들을 하나로 묶어놓은 "설계도"예요.
# - 예를 들어, "자동차" 클래스에는 "시동걸기", "전진", "후진" 같은 기능이 있죠.
# - 우리의 "Face3DScanner" 클래스에는 "사진찍기", "3D만들기" 같은 기능이 있어요.

class Face3DScanner:
    """
    얼굴 3D 스캐너 클래스
    
    이 클래스가 하는 일:
    1. 맥북 카메라를 켠다
    2. 사용자가 얼굴을 여러 각도로 돌리며 사진을 찍는다
    3. 찍은 사진들에서 특징점을 찾는다
    4. 특징점들로 3D 점구름(Point Cloud)을 만든다
    5. 점구름을 3D 모델 파일로 저장한다
    """
    
    # =========================================================================
    # 3-1: 초기화 함수 (__init__)
    # =========================================================================
    #
    # __init__은 "초기화 함수"예요.
    # 클래스로 객체를 만들 때 가장 먼저 실행되는 함수입니다.
    # 필요한 준비물들을 여기서 셋팅해요.
    
    def __init__(self):
        """
        스캐너 초기화
        - 필요한 변수들을 준비합니다
        - 얼굴 인식기를 로드합니다
        """
        print("=" * 60)
        print("🎭 얼굴 3D 스캐너를 초기화합니다...")
        print("=" * 60)
        
        # 찍은 사진들을 저장할 리스트 (빈 리스트로 시작)
        # 리스트는 여러 개의 데이터를 순서대로 담는 상자예요
        self.captured_images = []
        
        # 각 사진의 특징점들을 저장할 리스트
        self.all_keypoints = []
        
        # 3D 점들을 저장할 리스트
        self.points_3d = []
        
        # 얼굴 인식기 로드
        # CascadeClassifier는 미리 학습된 얼굴 인식 모델이에요
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
        # 특징점 검출기 생성
        # ORB는 이미지에서 특별한 점(코너, 엣지 등)을 찾아주는 알고리즘이에요
        # nfeatures=1000은 "최대 1000개의 특징점을 찾아라"는 뜻
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        
        # 특징점 매칭기 생성
        # BFMatcher는 두 이미지의 특징점을 비교해서 같은 점을 찾아주는 도구예요
        # NORM_HAMMING은 ORB 특징점 비교에 적합한 거리 측정 방식이에요
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 사진 저장 폴더 생성
        self._create_photo_folder()
        
        print("✅ 초기화 완료!")
        print()
    
    # =========================================================================
    # 3-2: 폴더 생성 함수
    # =========================================================================
    
    def _create_photo_folder(self):
        """
        사진을 저장할 폴더를 만듭니다.
        
        함수 이름 앞에 _가 붙으면 "내부용 함수"라는 관례적 표시예요.
        클래스 밖에서는 직접 호출하지 않고, 클래스 내부에서만 사용해요.
        """
        # 현재 시간을 폴더 이름에 넣어서 매번 새 폴더를 만들어요
        # 이렇게 하면 여러 번 실행해도 사진이 섞이지 않아요
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_folder = f"{PHOTO_FOLDER}_{timestamp}"
        
        # os.makedirs: 폴더를 만드는 함수
        # exist_ok=True: 이미 폴더가 있어도 에러 안 내고 넘어가기
        os.makedirs(self.save_folder, exist_ok=True)
        
        print(f"📁 사진 저장 폴더: {self.save_folder}")
    
    # =========================================================================
    # 3-3: 카메라 열기 함수
    # =========================================================================
    
    def open_camera(self):
        """
        맥북의 내장 카메라를 엽니다.
        
        Returns:
            camera: 카메라 객체 (성공시) 또는 None (실패시)
        
        Returns가 뭔가요?
        - 함수가 "돌려주는 값"이에요.
        - 예: 자판기에 돈을 넣으면 음료가 "반환"되죠? 그것처럼요.
        """
        print("📷 카메라를 여는 중...")
        
        # cv2.VideoCapture(0): 0번 카메라(내장 카메라)를 열어요
        # 외장 카메라를 쓰려면 숫자를 1, 2 등으로 바꾸면 돼요
        camera = cv2.VideoCapture(0)
        
        # 카메라가 제대로 열렸는지 확인
        if not camera.isOpened():
            print("❌ 에러: 카메라를 열 수 없습니다!")
            print("   - 카메라 권한을 확인해주세요")
            print("   - 시스템 환경설정 > 보안 및 개인정보 > 카메라")
            return None
        
        # 카메라 해상도 설정
        # CAP_PROP_FRAME_WIDTH: 가로 해상도 설정
        # CAP_PROP_FRAME_HEIGHT: 세로 해상도 설정
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        print("✅ 카메라 열기 성공!")
        return camera
    
    # =========================================================================
    # 3-4: 얼굴 인식 함수
    # =========================================================================
    
    def detect_face(self, frame):
        """
        이미지에서 얼굴을 찾습니다.
        
        Args:
            frame: 카메라에서 가져온 이미지 (numpy 배열)
        
        Returns:
            faces: 찾은 얼굴들의 위치 정보 (x, y, 너비, 높이)
        
        Args가 뭔가요?
        - 함수에 "넣어주는 값"이에요.
        - 예: 자판기에 "돈"을 넣잖아요? 그게 Args예요.
        """
        # 컬러 이미지를 흑백으로 변환
        # 왜? 얼굴 인식은 흑백 이미지에서 더 빠르고 정확해요
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 찾기
        # detectMultiScale: 이미지에서 얼굴을 찾는 함수
        # scaleFactor=1.1: 이미지를 10%씩 줄여가며 찾기 (다양한 크기의 얼굴 찾기)
        # minNeighbors=5: 최소 5번 이상 얼굴로 인식되어야 진짜 얼굴로 판단
        # minSize: 이것보다 작은 건 얼굴로 안 봄
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        return faces
    
    # =========================================================================
    # 3-5: 화면에 안내 표시하기
    # =========================================================================
    
    def draw_guide(self, frame, faces, photo_count):
        """
        카메라 화면에 안내선과 정보를 그립니다.
        
        Args:
            frame: 카메라 이미지
            faces: 찾은 얼굴들
            photo_count: 현재까지 찍은 사진 수
        """
        height, width = frame.shape[:2]  # 이미지의 높이, 너비 가져오기
        
        # 1. 중앙 안내 원 그리기
        # 이 원 안에 얼굴을 맞추면 좋은 사진을 찍을 수 있어요
        center = (width // 2, height // 2)  # //는 나눗셈 후 정수로 만들기
        cv2.circle(frame, center, 150, (0, 255, 0), 2)  # 녹색 원
        
        # 2. 찾은 얼굴에 사각형 그리기
        for (x, y, w, h) in faces:
            # rectangle: 사각형 그리기
            # (x, y): 왼쪽 위 꼭지점
            # (x+w, y+h): 오른쪽 아래 꼭지점
            # (0, 255, 0): 색상 (BGR 순서, 녹색)
            # 2: 선 두께
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 얼굴 위에 "Face Detected" 텍스트 표시
            cv2.putText(frame, "Face Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. 상단에 안내 메시지 표시
        instructions = [
            f"Photos: {photo_count}/{MAX_PHOTOS}",
            "Press SPACE to capture",
            "Press 'q' to finish",
            "Rotate your face slowly"
        ]
        
        # 각 메시지를 화면에 표시
        for i, text in enumerate(instructions):
            y_position = 30 + (i * 30)  # 줄마다 30픽셀씩 아래로
            cv2.putText(frame, text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. 각도 안내 (어떤 방향으로 얼굴을 돌릴지)
        if photo_count < MAX_PHOTOS:
            angle_guide = self._get_angle_guide(photo_count)
            cv2.putText(frame, angle_guide, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def _get_angle_guide(self, photo_count):
        """
        현재 몇 번째 사진인지에 따라 어떤 각도로 찍을지 안내합니다.
        """
        # 각도 안내 순서
        # 다양한 각도의 사진을 찍어야 3D 모델이 잘 만들어져요
        guides = [
            "1. Look straight at camera (정면)",
            "2. Turn head slightly LEFT (약간 왼쪽)",
            "3. Turn head more LEFT (더 왼쪽)",
            "4. Turn head slightly RIGHT (약간 오른쪽)",
            "5. Turn head more RIGHT (더 오른쪽)",
            "6. Tilt head UP slightly (약간 위)",
            "7. Tilt head DOWN slightly (약간 아래)",
            "8. Turn LEFT + tilt UP (왼쪽+위)",
            "9. Turn RIGHT + tilt UP (오른쪽+위)",
            "10. Turn LEFT + tilt DOWN (왼쪽+아래)",
            "11-20. Free angles (자유롭게)"
        ]
        
        if photo_count < len(guides) - 1:
            return guides[photo_count]
        else:
            return guides[-1]  # 마지막 안내 반복
    
    # =========================================================================
    # 3-6: 사진 캡처 함수
    # =========================================================================
    
    def capture_photo(self, frame, photo_count):
        """
        현재 화면을 사진으로 저장합니다.
        
        Args:
            frame: 저장할 이미지
            photo_count: 사진 번호
        
        Returns:
            저장된 파일 경로
        """
        # 파일 이름 생성 (예: face_001.jpg)
        filename = f"face_{photo_count:03d}.jpg"  # :03d는 3자리 숫자로 만들기
        filepath = os.path.join(self.save_folder, filename)
        
        # 이미지 저장
        # imwrite: 이미지를 파일로 저장하는 함수
        cv2.imwrite(filepath, frame)
        
        # 리스트에 이미지 추가
        self.captured_images.append(frame.copy())
        
        print(f"📸 사진 {photo_count} 저장 완료: {filename}")
        
        return filepath
    
    # =========================================================================
    # 3-7: 특징점 추출 함수
    # =========================================================================
    
    def extract_features(self, image):
        """
        이미지에서 특징점을 찾습니다.
        
        특징점(Feature Point / Keypoint)이란?
        - 이미지에서 "특별한" 점들이에요
        - 예: 코너, 엣지, 점 등
        - 사람으로 치면 "눈", "코", "입꼬리" 같은 거예요
        - 이 점들을 여러 사진에서 찾아서 비교하면 3D 위치를 알 수 있어요
        
        Args:
            image: 분석할 이미지
        
        Returns:
            keypoints: 찾은 특징점들
            descriptors: 각 특징점의 "설명서" (특징점을 구별하는 데 사용)
        """
        # 흑백으로 변환 (특징점 검출은 흑백에서 해요)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 특징점 찾기
        # detectAndCompute: 특징점을 찾고, 각 특징점의 설명(descriptor)도 계산
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        print(f"   찾은 특징점 수: {len(keypoints)}")
        
        return keypoints, descriptors
    
    # =========================================================================
    # 3-8: 특징점 매칭 함수
    # =========================================================================
    
    def match_features(self, desc1, desc2):
        """
        두 이미지의 특징점을 비교해서 같은 점을 찾습니다.
        
        왜 필요한가요?
        - 사진1의 "코 끝"과 사진2의 "코 끝"이 같은 점인지 알아야
        - 그 점의 3D 위치를 계산할 수 있어요
        
        Args:
            desc1: 첫 번째 이미지의 특징점 설명서들
            desc2: 두 번째 이미지의 특징점 설명서들
        
        Returns:
            matches: 매칭된 점들의 쌍
        """
        if desc1 is None or desc2 is None:
            return []
        
        # 매칭 수행
        matches = self.matcher.match(desc1, desc2)
        
        # 거리순으로 정렬 (거리가 가까울수록 좋은 매칭)
        # 거리 = 두 특징점이 얼마나 비슷한지 (작을수록 비슷)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 상위 50%만 사용 (좋은 매칭만 선택)
        good_matches = matches[:len(matches)//2]
        
        return good_matches
    
    # =========================================================================
    # 3-9: 3D 점 계산 함수 (핵심!)
    # =========================================================================
    
    def calculate_3d_points(self):
        """
        여러 사진에서 찾은 특징점들로 3D 점구름을 만듭니다.
        
        점구름(Point Cloud)이란?
        - 3D 공간에 점들이 구름처럼 모여있는 것
        - 이 점들을 연결하면 3D 모델이 됩니다
        
        이 함수의 원리 (삼각측량):
        - 두 눈으로 물체를 보면 거리를 알 수 있죠?
        - 그것처럼 여러 각도의 사진으로 점의 3D 위치를 계산해요
        """
        print("\n" + "=" * 60)
        print("🔍 특징점 분석 및 3D 점 계산 중...")
        print("=" * 60)
        
        if len(self.captured_images) < 2:
            print("❌ 최소 2장의 사진이 필요합니다!")
            return []
        
        all_3d_points = []
        all_colors = []
        
        # 모든 이미지에서 특징점 추출
        print("\n📌 각 사진에서 특징점 추출 중...")
        features_list = []
        for i, img in enumerate(self.captured_images):
            print(f"   사진 {i+1}/{len(self.captured_images)} 분석 중...")
            kp, desc = self.extract_features(img)
            features_list.append((kp, desc, img))
        
        # 연속된 이미지 쌍에서 3D 점 계산
        print("\n🧮 3D 좌표 계산 중...")
        for i in range(len(features_list) - 1):
            kp1, desc1, img1 = features_list[i]
            kp2, desc2, img2 = features_list[i + 1]
            
            print(f"   이미지 쌍 {i+1}-{i+2} 처리 중...")
            
            # 특징점 매칭
            matches = self.match_features(desc1, desc2)
            
            if len(matches) < 10:
                print(f"   ⚠️ 매칭점 부족 (찾은 개수: {len(matches)})")
                continue
            
            print(f"   ✅ 매칭된 특징점: {len(matches)}개")
            
            # 매칭된 점들의 좌표 추출
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # 간단한 3D 점 생성 (깊이 추정)
            # 실제로는 카메라 보정(calibration)이 필요하지만,
            # 여기서는 간단히 시차(disparity)를 이용해 깊이를 추정해요
            for j, (p1, p2) in enumerate(zip(pts1, pts2)):
                # 시차 계산 (두 점의 x좌표 차이)
                disparity = abs(p1[0] - p2[0])
                
                # 깊이 계산 (시차가 클수록 가까움)
                # 0으로 나누기 방지
                if disparity > 1:
                    depth = 1000.0 / disparity  # 간단한 깊이 추정
                else:
                    depth = 500.0
                
                # 3D 점 좌표 (x, y, z)
                x = (p1[0] - CAMERA_WIDTH/2) / 10  # 중앙 기준으로 변환
                y = (p1[1] - CAMERA_HEIGHT/2) / 10
                z = depth
                
                all_3d_points.append([x, y, z])
                
                # 해당 위치의 색상 가져오기
                px, py = int(p1[0]), int(p1[1])
                if 0 <= px < img1.shape[1] and 0 <= py < img1.shape[0]:
                    color = img1[py, px] / 255.0  # 0-1 범위로 정규화
                    all_colors.append(color[::-1])  # BGR -> RGB
                else:
                    all_colors.append([0.5, 0.5, 0.5])  # 기본 회색
        
        self.points_3d = np.array(all_3d_points)
        self.colors = np.array(all_colors)
        
        print(f"\n✅ 총 {len(self.points_3d)}개의 3D 점 생성 완료!")
        
        return self.points_3d
    
    # =========================================================================
    # 3-10: 3D 모델 저장 함수
    # =========================================================================
    
    def save_3d_model(self):
        """
        3D 점구름을 파일로 저장합니다.
        
        저장 형식:
        - PLY: 점구름 형식 (대부분의 3D 뷰어에서 열 수 있음)
        - OBJ: 메쉬 형식 (Blender 등에서 편집 가능)
        """
        print("\n" + "=" * 60)
        print("💾 3D 모델 저장 중...")
        print("=" * 60)
        
        if len(self.points_3d) == 0:
            print("❌ 저장할 3D 점이 없습니다!")
            return None
        
        try:
            import open3d as o3d
            
            # Open3D 점구름 객체 생성
            pcd = o3d.geometry.PointCloud()
            
            # 점 좌표 설정
            pcd.points = o3d.utility.Vector3dVector(self.points_3d)
            
            # 색상 설정
            if len(self.colors) == len(self.points_3d):
                pcd.colors = o3d.utility.Vector3dVector(self.colors)
            
            # 노이즈 제거 (이상한 점들 삭제)
            print("🧹 노이즈 제거 중...")
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # PLY 파일로 저장
            ply_path = os.path.join(self.save_folder, "face_3d_model.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"✅ PLY 파일 저장: {ply_path}")
            
            # 점구름을 메쉬로 변환 시도
            print("🔷 메쉬 생성 중...")
            try:
                # 법선 벡터 계산 (메쉬 생성에 필요)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=10, max_nn=30
                    )
                )
                
                # Poisson 재구성으로 메쉬 생성
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8
                )
                
                # OBJ 파일로 저장
                obj_path = os.path.join(self.save_folder, "face_3d_model.obj")
                o3d.io.write_triangle_mesh(obj_path, mesh)
                print(f"✅ OBJ 파일 저장: {obj_path}")
                
            except Exception as e:
                print(f"⚠️ 메쉬 생성 실패: {e}")
                print("   점구름(PLY) 파일은 정상 저장되었습니다.")
            
            return ply_path
            
        except ImportError:
            print("⚠️ Open3D가 설치되지 않았습니다.")
            print("   pip3 install open3d 로 설치해주세요.")
            
            # Open3D 없이 간단한 PLY 파일 생성
            return self._save_simple_ply()
    
    def _save_simple_ply(self):
        """
        Open3D 없이 간단한 PLY 파일을 생성합니다.
        """
        ply_path = os.path.join(self.save_folder, "face_3d_model.ply")
        
        with open(ply_path, 'w') as f:
            # PLY 헤더 작성
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 점 데이터 작성
            for i, point in enumerate(self.points_3d):
                if i < len(self.colors):
                    r, g, b = (self.colors[i] * 255).astype(int)
                else:
                    r, g, b = 128, 128, 128
                f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
        
        print(f"✅ 간단한 PLY 파일 저장: {ply_path}")
        return ply_path
    
    # =========================================================================
    # 3-11: 메인 실행 함수
    # =========================================================================
    
    def run(self):
        """
        스캐너를 실행하는 메인 함수입니다.
        
        실행 순서:
        1. 카메라 열기
        2. 사진 촬영 (사용자 조작)
        3. 3D 점 계산
        4. 3D 모델 저장
        """
        print("\n" + "=" * 60)
        print("🚀 얼굴 3D 스캐너 시작!")
        print("=" * 60)
        print("\n📋 사용 방법:")
        print("   - SPACE: 사진 촬영")
        print("   - Q: 촬영 종료 및 3D 모델 생성")
        print("   - ESC: 프로그램 종료")
        print("\n💡 팁: 얼굴을 천천히 돌려가며 다양한 각도에서 찍으세요!")
        print()
        
        # 카메라 열기
        camera = self.open_camera()
        if camera is None:
            return
        
        photo_count = 0
        countdown_active = False
        countdown_start = 0
        
        print("\n📷 카메라가 열렸습니다. 촬영을 시작하세요!")
        
        try:
            while True:
                # 카메라에서 프레임 읽기
                ret, frame = camera.read()
                
                if not ret:
                    print("❌ 카메라에서 영상을 읽을 수 없습니다.")
                    break
                
                # 좌우 반전 (거울처럼 보이게)
                frame = cv2.flip(frame, 1)
                
                # 얼굴 인식
                faces = self.detect_face(frame)
                
                # 화면에 안내 표시
                display_frame = self.draw_guide(frame.copy(), faces, photo_count)
                
                # 카운트다운 처리
                if countdown_active:
                    elapsed = time.time() - countdown_start
                    remaining = 3 - int(elapsed)
                    
                    if remaining > 0:
                        # 카운트다운 숫자 표시
                        cv2.putText(display_frame, str(remaining), 
                                   (CAMERA_WIDTH//2 - 30, CAMERA_HEIGHT//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                    else:
                        # 촬영!
                        photo_count += 1
                        self.capture_photo(frame, photo_count)
                        countdown_active = False
                        
                        # 최대 사진 수 도달
                        if photo_count >= MAX_PHOTOS:
                            print(f"\n✅ {MAX_PHOTOS}장 촬영 완료!")
                            break
                
                # 화면 표시
                cv2.imshow('Face 3D Scanner', display_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE: 촬영
                    if not countdown_active and len(faces) > 0:
                        countdown_active = True
                        countdown_start = time.time()
                        print("📸 3초 후 촬영...")
                    elif len(faces) == 0:
                        print("⚠️ 얼굴이 인식되지 않습니다. 카메라를 바라봐주세요.")
                
                elif key == ord('q'):  # Q: 종료
                    if photo_count >= MIN_PHOTOS:
                        print(f"\n✅ 촬영 종료 ({photo_count}장)")
                        break
                    else:
                        print(f"⚠️ 최소 {MIN_PHOTOS}장이 필요합니다. "
                              f"현재: {photo_count}장")
                
                elif key == 27:  # ESC: 프로그램 완전 종료
                    print("\n👋 프로그램을 종료합니다.")
                    camera.release()
                    cv2.destroyAllWindows()
                    return
        
        finally:
            # 카메라 닫기
            camera.release()
            cv2.destroyAllWindows()
        
        # 충분한 사진이 있으면 3D 모델 생성
        if photo_count >= MIN_PHOTOS:
            # 3D 점 계산
            self.calculate_3d_points()
            
            # 3D 모델 저장
            model_path = self.save_3d_model()
            
            # 결과 출력
            print("\n" + "=" * 60)
            print("🎉 3D 스캔 완료!")
            print("=" * 60)
            print(f"📁 저장 위치: {self.save_folder}")
            print(f"📷 촬영 사진: {photo_count}장")
            print(f"📍 생성된 3D 점: {len(self.points_3d)}개")
            print("\n📂 생성된 파일:")
            print(f"   - 사진들: face_001.jpg ~ face_{photo_count:03d}.jpg")
            print(f"   - 3D 모델: face_3d_model.ply")
            print("\n💡 PLY 파일은 다음 프로그램으로 열 수 있어요:")
            print("   - MeshLab (무료): https://www.meshlab.net/")
            print("   - Blender (무료): https://www.blender.org/")
            print("   - macOS Preview (기본 앱)")
        else:
            print(f"\n⚠️ 사진이 부족합니다 ({photo_count}장)")
            print(f"   최소 {MIN_PHOTOS}장이 필요해요.")


# =============================================================================
# 4부: 프로그램 실행
# =============================================================================
#
# if __name__ == "__main__": 이란?
# - 이 파일을 직접 실행할 때만 아래 코드가 실행돼요
# - 다른 파일에서 import 할 때는 실행되지 않아요
# - 관례적으로 파이썬 프로그램의 시작점을 표시해요

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            🎭 얼굴 3D 스캐너 프로그램 🎭                      ║
    ║                                                               ║
    ║     맥북 카메라로 얼굴을 찍어 3D 모델을 만들어보세요!         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 스캐너 객체 생성 및 실행
    scanner = Face3DScanner()
    scanner.run()