"""
=============================================================================
🎭 얼굴 3D 스캐너 - 올인원 자동화 버전
=============================================================================

실행하면:
1. 카메라 자동 실행
2. 10장 자동/수동 촬영
3. 3D 합성
4. 바로 3D 뷰어로 결과 확인!

사용법:
    python3 face_3d_auto.py

조작:
    SPACE: 사진 촬영
    A: 자동 촬영 모드 (2초 간격으로 10장 자동 촬영)
    Q: 촬영 종료 후 3D 생성
    ESC: 프로그램 종료

=============================================================================
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# Open3D 체크
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️ Open3D 미설치! 설치하세요: pip3 install open3d")
    print("   (3D 뷰어 기능이 필요합니다)\n")


class Face3DAutoScanner:
    """
    자동화된 얼굴 3D 스캐너
    - 10장 촬영 후 자동으로 3D 생성 및 뷰어 실행
    """
    
    def __init__(self):
        # 설정
        self.TOTAL_PHOTOS = 10          # 찍을 사진 수
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        self.AUTO_INTERVAL = 2.0        # 자동 촬영 간격 (초)
        
        # 데이터 저장용
        self.images = []
        self.points_3d = None
        self.colors = None
        self.point_cloud = None
        
        # 저장 폴더
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_folder = f"face_scan_{timestamp}"
        os.makedirs(self.save_folder, exist_ok=True)
        
        # 얼굴 인식기
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 특징점 검출기
        try:
            self.detector = cv2.SIFT_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        except:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        print("=" * 50)
        print("🎭 얼굴 3D 스캐너 준비 완료!")
        print("=" * 50)
    
    def run(self):
        """메인 실행 - 촬영부터 3D 뷰어까지 한번에!"""
        
        print("\n📋 조작법:")
        print("   SPACE : 사진 촬영")
        print("   A     : 자동 촬영 (2초 간격)")
        print("   Q     : 촬영 완료 → 3D 생성")
        print("   ESC   : 종료\n")
        
        # 1단계: 사진 촬영
        success = self._capture_photos()
        if not success:
            return
        
        # 2단계: 3D 점구름 생성
        self._create_3d_points()
        
        # 3단계: 파일 저장
        self._save_files()
        
        # 4단계: 3D 뷰어 실행
        self._show_3d_viewer()
        
        print("\n✅ 모든 작업 완료!")
        print(f"📁 저장 위치: {self.save_folder}")
    
    # =========================================================================
    # 1단계: 사진 촬영
    # =========================================================================
    
    def _capture_photos(self):
        """카메라로 사진 촬영"""
        
        print("📷 카메라 시작...")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("❌ 카메라를 열 수 없습니다!")
            print("   시스템 설정 > 개인정보 보호 > 카메라 권한 확인")
            return False
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        
        photo_count = 0
        auto_mode = False
        last_auto_time = 0
        
        # 각도 안내
        angles = [
            "1/10: Front (정면)",
            "2/10: Left 15°",
            "3/10: Left 30°",
            "4/10: Right 15°",
            "5/10: Right 30°",
            "6/10: Up (위)",
            "7/10: Down (아래)",
            "8/10: Left+Up",
            "9/10: Right+Up",
            "10/10: Free"
        ]
        
        print("✅ 카메라 준비 완료! 촬영을 시작하세요.\n")
        
        while photo_count < self.TOTAL_PHOTOS:
            ret, frame = camera.read()
            if not ret:
                break
            
            # 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            # 얼굴 인식
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # 화면에 정보 표시
            h, w = display.shape[:2]
            
            # 중앙 가이드 원
            cv2.circle(display, (w//2, h//2), 150, (0, 255, 0), 2)
            
            # 얼굴 표시
            face_detected = len(faces) > 0
            for (x, y, fw, fh) in faces:
                cv2.rectangle(display, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            
            # 상태 표시
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            status_text = "Face OK" if face_detected else "No Face!"
            
            cv2.putText(display, f"Photos: {photo_count}/{self.TOTAL_PHOTOS}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, status_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 모드 표시
            mode_text = "AUTO MODE (2sec)" if auto_mode else "MANUAL (SPACE to capture)"
            cv2.putText(display, mode_text, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 각도 안내
            if photo_count < len(angles):
                cv2.putText(display, angles[photo_count], 
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 자동 촬영 모드
            if auto_mode and face_detected:
                current_time = time.time()
                if current_time - last_auto_time >= self.AUTO_INTERVAL:
                    # 촬영!
                    photo_count += 1
                    self._save_photo(frame, photo_count)
                    last_auto_time = current_time
                    
                    # 플래시 효과
                    cv2.rectangle(display, (0, 0), (w, h), (255, 255, 255), -1)
                else:
                    # 카운트다운 표시
                    remaining = self.AUTO_INTERVAL - (current_time - last_auto_time)
                    cv2.putText(display, f"Next: {remaining:.1f}s", 
                               (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow("Face 3D Scanner", display)
            
            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and face_detected and not auto_mode:
                # 수동 촬영
                photo_count += 1
                self._save_photo(frame, photo_count)
                print(f"📸 촬영 {photo_count}/{self.TOTAL_PHOTOS}")
                
            elif key == ord('a'):
                # 자동 모드 토글
                auto_mode = not auto_mode
                last_auto_time = time.time()
                mode = "ON" if auto_mode else "OFF"
                print(f"🔄 자동 촬영 모드: {mode}")
                
            elif key == ord('q'):
                if photo_count >= 5:  # 최소 5장
                    break
                else:
                    print(f"⚠️ 최소 5장 필요! 현재: {photo_count}장")
                    
            elif key == 27:  # ESC
                camera.release()
                cv2.destroyAllWindows()
                print("👋 종료")
                return False
        
        camera.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ 촬영 완료! ({photo_count}장)")
        return photo_count >= 5
    
    def _save_photo(self, frame, count):
        """사진 저장"""
        filepath = os.path.join(self.save_folder, f"photo_{count:02d}.jpg")
        cv2.imwrite(filepath, frame)
        self.images.append(frame.copy())
    
    # =========================================================================
    # 2단계: 3D 점구름 생성
    # =========================================================================
    
    def _create_3d_points(self):
        """사진들에서 3D 점구름 생성"""
        
        print("\n🔧 3D 모델 생성 중...")
        print("=" * 50)
        
        if len(self.images) < 2:
            print("❌ 이미지 부족!")
            return
        
        all_points = []
        all_colors = []
        
        # 카메라 파라미터 (근사값)
        focal = self.CAMERA_WIDTH * 0.8
        cx, cy = self.CAMERA_WIDTH / 2, self.CAMERA_HEIGHT / 2
        K = np.array([[focal, 0, cx],
                      [0, focal, cy],
                      [0, 0, 1]], dtype=np.float32)
        
        # 특징점 추출
        print("📌 특징점 추출 중...")
        features = []
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detectAndCompute(gray, None)
            features.append((kp, desc, img))
            print(f"   이미지 {i+1}: {len(kp)}개 특징점")
        
        # 연속 이미지 쌍 매칭
        print("\n🔗 3D 좌표 계산 중...")
        for i in range(len(features) - 1):
            kp1, desc1, img1 = features[i]
            kp2, desc2, img2 = features[i + 1]
            
            if desc1 is None or desc2 is None:
                continue
            
            # 매칭
            try:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            except:
                matches = self.matcher.match(desc1, desc2)
                good = sorted(matches, key=lambda x: x.distance)[:len(matches)//2]
            
            if len(good) < 10:
                continue
            
            print(f"   쌍 {i+1}-{i+2}: {len(good)}개 매칭")
            
            # 매칭 좌표
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            
            # Essential Matrix & 삼각측량
            try:
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                    
                    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                    P2 = K @ np.hstack([R, t])
                    
                    pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
                    pts2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)
                    
                    points_4d = cv2.triangulatePoints(P1, P2, 
                                                       pts1_undist.reshape(-1, 2).T,
                                                       pts2_undist.reshape(-1, 2).T)
                    points_3d = (points_4d[:3] / points_4d[3]).T
                    
                    # 유효한 점 필터링
                    for j, pt in enumerate(points_3d):
                        if -200 < pt[0] < 200 and -200 < pt[1] < 200 and 0.1 < pt[2] < 500:
                            all_points.append(pt * 50)  # 스케일 조정
                            
                            px, py = int(pts1[j][0]), int(pts1[j][1])
                            if 0 <= px < img1.shape[1] and 0 <= py < img1.shape[0]:
                                c = img1[py, px] / 255.0
                                all_colors.append([c[2], c[1], c[0]])
                            else:
                                all_colors.append([0.5, 0.5, 0.5])
            except:
                # 폴백: 간단한 시차 기반
                for j, (p1, p2) in enumerate(zip(pts1, pts2)):
                    disp = np.linalg.norm(p1 - p2)
                    if disp > 1:
                        z = 5000 / disp
                        if 10 < z < 500:
                            x = (p1[0] - cx) * z / focal
                            y = (p1[1] - cy) * z / focal
                            all_points.append([x, y, z])
                            
                            px, py = int(p1[0]), int(p1[1])
                            if 0 <= px < img1.shape[1] and 0 <= py < img1.shape[0]:
                                c = img1[py, px] / 255.0
                                all_colors.append([c[2], c[1], c[0]])
                            else:
                                all_colors.append([0.5, 0.5, 0.5])
        
        if len(all_points) == 0:
            print("❌ 3D 점 생성 실패!")
            return
        
        self.points_3d = np.array(all_points)
        self.colors = np.array(all_colors)
        
        print(f"\n✅ {len(self.points_3d)}개 3D 점 생성!")
        
        # Open3D 점구름 생성
        if HAS_OPEN3D:
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(self.points_3d)
            self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
            
            # 노이즈 제거
            print("🧹 노이즈 제거...")
            self.point_cloud, _ = self.point_cloud.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            print(f"   → {len(self.point_cloud.points)}개 점 (정제 후)")
    
    # =========================================================================
    # 3단계: 파일 저장
    # =========================================================================
    
    def _save_files(self):
        """3D 모델 파일 저장"""
        
        print("\n💾 파일 저장 중...")
        
        if self.point_cloud is None and self.points_3d is None:
            print("❌ 저장할 데이터 없음!")
            return
        
        # PLY 저장
        ply_path = os.path.join(self.save_folder, "face_3d.ply")
        if HAS_OPEN3D and self.point_cloud is not None:
            o3d.io.write_point_cloud(ply_path, self.point_cloud)
        else:
            self._save_ply_manual(ply_path)
        print(f"   ✅ {ply_path}")
        
        # OBJ 저장
        obj_path = os.path.join(self.save_folder, "face_3d.obj")
        self._save_obj(obj_path)
        print(f"   ✅ {obj_path}")
        
        # 메쉬 생성 시도 (STL)
        if HAS_OPEN3D and self.point_cloud is not None:
            try:
                stl_path = os.path.join(self.save_folder, "face_3d.stl")
                
                # 법선 추정
                self.point_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
                )
                
                # Poisson 메쉬 생성
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.point_cloud, depth=8
                )
                
                o3d.io.write_triangle_mesh(stl_path, mesh)
                print(f"   ✅ {stl_path} (메쉬)")
            except Exception as e:
                print(f"   ⚠️ STL 생성 실패: {e}")
    
    def _save_ply_manual(self, filepath):
        """PLY 수동 저장"""
        with open(filepath, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            
            for i, pt in enumerate(self.points_3d):
                r, g, b = (self.colors[i] * 255).astype(int) if i < len(self.colors) else (128, 128, 128)
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {r} {g} {b}\n")
    
    def _save_obj(self, filepath):
        """OBJ 저장"""
        pts = np.asarray(self.point_cloud.points) if HAS_OPEN3D and self.point_cloud else self.points_3d
        cols = np.asarray(self.point_cloud.colors) if HAS_OPEN3D and self.point_cloud else self.colors
        
        with open(filepath, 'w') as f:
            f.write("# Face 3D Model\n")
            for i, pt in enumerate(pts):
                r, g, b = cols[i] if i < len(cols) else (0.5, 0.5, 0.5)
                f.write(f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")
    
    # =========================================================================
    # 4단계: 3D 뷰어
    # =========================================================================
    
    def _show_3d_viewer(self):
        """Open3D 뷰어로 결과 표시"""
        
        if not HAS_OPEN3D:
            print("\n⚠️ 3D 뷰어를 보려면 Open3D를 설치하세요:")
            print("   pip3 install open3d")
            return
        
        if self.point_cloud is None or len(self.point_cloud.points) == 0:
            print("❌ 표시할 3D 데이터 없음!")
            return
        
        print("\n" + "=" * 50)
        print("🖥️  3D 뷰어 실행!")
        print("=" * 50)
        print("\n📋 뷰어 조작법:")
        print("   마우스 왼쪽 드래그 : 회전")
        print("   마우스 휠         : 확대/축소")
        print("   마우스 오른쪽 드래그: 이동")
        print("   Q / ESC           : 뷰어 종료")
        print("\n🎨 3D 뷰어가 새 창에서 열립니다...\n")
        
        # 좌표축 생성
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=30, origin=[0, 0, 0]
        )
        
        # 뷰어 실행
        o3d.visualization.draw_geometries(
            [self.point_cloud, coord_frame],
            window_name="🎭 Face 3D Model",
            width=1200,
            height=800,
            point_show_normal=False
        )
        
        print("✅ 뷰어 종료")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          🎭 얼굴 3D 스캐너 - 올인원 버전 🎭              ║
    ║                                                          ║
    ║     실행 → 10장 촬영 → 3D 생성 → 바로 뷰어로 확인!      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Python 버전 체크
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Python {py_version}")
    
    if not HAS_OPEN3D:
        print("\n⚠️  Open3D가 필요합니다!")
        print("    설치: pip3 install open3d")
        print("    (Python 3.9 이상 권장)\n")
        
        response = input("Open3D 없이 계속할까요? (파일만 저장됨) [y/n]: ").strip().lower()
        if response != 'y':
            print("👋 종료")
            sys.exit(0)
    
    print()
    
    # 스캐너 실행
    scanner = Face3DAutoScanner()
    scanner.run()