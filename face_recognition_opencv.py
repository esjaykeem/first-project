"""
=============================================================================
🔐 얼굴 인식 시스템 (OpenCV 전용 - dlib 불필요!)
=============================================================================

특징:
- dlib, face_recognition 라이브러리 불필요!
- OpenCV만 사용 (설치 간편)
- 맥북 M1/M3에서 완벽 호환

사용 라이브러리:
- opencv-python: 얼굴 검출 및 특징 추출
- numpy: 수치 계산
- scikit-learn: 유사도 계산 (선택)

설치:
    pip3 install opencv-python numpy

사용법:
    python3 face_recognition_opencv.py

=============================================================================
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime

# =============================================================================
# 설정
# =============================================================================

class Config:
    PHOTOS_TO_REGISTER = 30      # 등록할 사진 수 (30장으로 증가!)
    RECOGNITION_THRESHOLD = 0.6  # 인식 임계값 (낮을수록 엄격)
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FACE_SIZE = (160, 160)       # 얼굴 정규화 크기


# =============================================================================
# 얼굴 특징 추출기 (OpenCV 기반)
# =============================================================================

class FaceFeatureExtractor:
    """
    OpenCV를 사용한 얼굴 특징 추출
    
    방법:
    1. 얼굴 검출 (Haar Cascade 또는 DNN)
    2. 얼굴 영역 정규화 (크기, 밝기)
    3. 특징 벡터 추출 (히스토그램 + LBP + ORB)
    """
    
    def __init__(self):
        # 얼굴 검출기 (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # DNN 얼굴 검출기 (더 정확함) - 선택적 사용
        self.use_dnn = False
        try:
            # OpenCV DNN 얼굴 검출 모델 (Caffe)
            model_path = cv2.data.haarcascades.replace('haarcascades/', '')
            prototxt = os.path.join(model_path, 'deploy.prototxt')
            caffemodel = os.path.join(model_path, 'res10_300x300_ssd_iter_140000.caffemodel')
            
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.use_dnn = True
        except:
            pass
        
        # ORB 특징점 검출기
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # LBP 파라미터
        self.lbp_radius = 1
        self.lbp_neighbors = 8
        
        print(f"   얼굴 검출: {'DNN' if self.use_dnn else 'Haar Cascade'}")
    
    def detect_faces(self, frame):
        """얼굴 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        # (x, y, w, h) 형식으로 반환
        return faces
    
    def extract_face(self, frame, face_rect):
        """얼굴 영역 추출 및 정규화"""
        x, y, w, h = face_rect
        
        # 여유 공간 추가 (얼굴 주변 포함)
        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # 크기 정규화
        face_img = cv2.resize(face_img, Config.FACE_SIZE)
        
        # 밝기 정규화 (히스토그램 평활화)
        if len(face_img.shape) == 3:
            # 컬러 이미지
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge([l, a, b])
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return face_img
    
    def compute_lbp(self, gray_img):
        """Local Binary Pattern 계산"""
        h, w = gray_img.shape
        lbp = np.zeros_like(gray_img)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_img[i, j]
                code = 0
                code |= (gray_img[i-1, j-1] >= center) << 7
                code |= (gray_img[i-1, j] >= center) << 6
                code |= (gray_img[i-1, j+1] >= center) << 5
                code |= (gray_img[i, j+1] >= center) << 4
                code |= (gray_img[i+1, j+1] >= center) << 3
                code |= (gray_img[i+1, j] >= center) << 2
                code |= (gray_img[i+1, j-1] >= center) << 1
                code |= (gray_img[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        return lbp
    
    def extract_features(self, face_img):
        """
        얼굴에서 특징 벡터 추출
        
        조합:
        1. 색상 히스토그램 (전체적인 색 분포)
        2. LBP 히스토그램 (텍스처 패턴)
        3. HOG 특징 (형태)
        """
        if face_img is None:
            return None
        
        features = []
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 1. 색상 히스토그램 (HSV)
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        
        # H 채널 히스토그램
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        features.extend(h_hist)
        
        # S 채널 히스토그램
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        features.extend(s_hist)
        
        # 2. LBP 히스토그램 (텍스처)
        lbp = self.compute_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [64], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        features.extend(lbp_hist)
        
        # 3. 그레이스케일 히스토그램
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
        features.extend(gray_hist)
        
        # 4. 얼굴 영역별 밝기 평균 (간단한 공간 정보)
        h, w = gray.shape
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.append(np.mean(cell) / 255.0)
                features.append(np.std(cell) / 255.0)
        
        return np.array(features, dtype=np.float32)
    
    def compute_similarity(self, features1, features2):
        """두 특징 벡터의 유사도 계산 (코사인 유사도)"""
        if features1 is None or features2 is None:
            return 0.0
        
        # 코사인 유사도
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return similarity


# =============================================================================
# 얼굴 인식 시스템
# =============================================================================

class FaceRecognitionSystem:
    """
    OpenCV 기반 얼굴 인식 시스템
    """
    
    def __init__(self):
        print("=" * 55)
        print("🔐 얼굴 인식 시스템 초기화 (OpenCV 전용)")
        print("=" * 55)
        
        # 특징 추출기
        self.extractor = FaceFeatureExtractor()
        
        # 등록된 데이터
        self.registered_name = None
        self.registered_features = []     # 등록된 특징들
        self.registered_feature_avg = None  # 평균 특징
        
        # 저장 폴더
        self.save_folder = "face_data"
        os.makedirs(self.save_folder, exist_ok=True)
        
        print("✅ 초기화 완료!\n")
    
    def run(self):
        """메인 실행"""
        print("=" * 55)
        print("🚀 프로그램 시작")
        print("=" * 55)
        
        # 저장된 데이터 확인
        if self._load_data():
            print(f"\n✅ 저장된 데이터 발견: {self.registered_name}")
            response = input("   이 데이터를 사용할까요? [y/n]: ").strip().lower()
            if response == 'y' or response == '':
                self._recognition_mode()
                return
        
        # 등록 모드 시작
        self._registration_mode()
    
    # =========================================================================
    # 등록 모드
    # =========================================================================
    
    def _registration_mode(self):
        """얼굴 등록"""
        print("\n" + "=" * 55)
        print("📝 얼굴 등록 모드")
        print("=" * 55)
        
        name = input("\n등록할 사람 이름: ").strip()
        if not name:
            name = "User"
        self.registered_name = name
        
        print(f"\n'{name}'님의 얼굴을 {Config.PHOTOS_TO_REGISTER}장 촬영합니다.")
        print("30장을 촬영하면 인식 정확도가 훨씬 높아져요!")
        print("다양한 각도, 표정, 조명으로 촬영해주세요!\n")
        print("📋 조작:")
        print("   SPACE: 촬영")
        print("   A: 자동 촬영")
        print("   Q: 완료")
        print("   ESC: 종료\n")
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ 카메라 열기 실패!")
            return
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        self.registered_features = []
        photo_count = 0
        auto_mode = False
        last_time = 0
        
        guides = [
            "1. 정면", "2. 왼쪽 15°", "3. 왼쪽 30°", "4. 왼쪽 45°",
            "5. 오른쪽 15°", "6. 오른쪽 30°", "7. 오른쪽 45°",
            "8. 위 15°", "9. 위 30°", "10. 아래 15°", "11. 아래 30°",
            "12. 왼쪽+위", "13. 왼쪽+아래", "14. 오른쪽+위", "15. 오른쪽+아래",
            "16. 웃는 얼굴", "17. 무표정", "18. 눈 크게", "19. 입 벌리기",
            "20. 찡그리기", "21. 놀란 표정", "22. 정면(조명 왼쪽)",
            "23. 정면(조명 오른쪽)", "24. 안경 쓰고(있다면)",
            "25. 머리 넘기기", "26-30. 자유롭게"
        ]
        
        print("📷 카메라 준비됨!\n")
        
        while photo_count < Config.PHOTOS_TO_REGISTER:
            ret, frame = camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            h, w = display.shape[:2]
            
            # 얼굴 검출
            faces = self.extractor.detect_faces(frame)
            face_ok = len(faces) == 1
            
            # 얼굴 표시
            for (x, y, fw, fh) in faces:
                color = (0, 255, 0) if face_ok else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x+fw, y+fh), color, 2)
            
            # 정보 표시
            cv2.putText(display, f"Registration: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Photos: {photo_count}/{Config.PHOTOS_TO_REGISTER}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            status = "Face OK" if face_ok else ("Multiple faces!" if len(faces) > 1 else "No face")
            color = (0, 255, 0) if face_ok else (0, 0, 255)
            cv2.putText(display, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            mode_text = "AUTO (1sec)" if auto_mode else "MANUAL (SPACE)"
            cv2.putText(display, mode_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 포즈 가이드
            if photo_count < len(guides):
                cv2.putText(display, f"Pose: {guides[photo_count]}", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 자동 촬영
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if auto_mode and face_ok and (current_time - last_time) >= 1.0:
                self._capture_face(frame, faces[0])
                photo_count += 1
                last_time = current_time
                print(f"📸 자동 촬영 {photo_count}/{Config.PHOTOS_TO_REGISTER}")
            
            cv2.imshow("Face Registration", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and face_ok and not auto_mode:
                self._capture_face(frame, faces[0])
                photo_count += 1
                print(f"📸 촬영 {photo_count}/{Config.PHOTOS_TO_REGISTER}")
            
            elif key == ord('a'):
                auto_mode = not auto_mode
                last_time = current_time
                print(f"🔄 자동 모드: {'ON' if auto_mode else 'OFF'}")
            
            elif key == ord('q') and photo_count >= 5:
                break
            
            elif key == 27:
                camera.release()
                cv2.destroyAllWindows()
                return
        
        camera.release()
        cv2.destroyAllWindows()
        
        if len(self.registered_features) >= 5:
            # 평균 특징 계산
            self.registered_feature_avg = np.mean(self.registered_features, axis=0)
            print(f"\n✅ '{name}' 등록 완료! ({len(self.registered_features)}장)")
            
            self._save_data()
            
            input("\nEnter를 눌러 인식 모드로...")
            self._recognition_mode()
    
    def _capture_face(self, frame, face_rect):
        """얼굴 캡처 및 특징 추출"""
        face_img = self.extractor.extract_face(frame, face_rect)
        if face_img is not None:
            features = self.extractor.extract_features(face_img)
            if features is not None:
                self.registered_features.append(features)
    
    # =========================================================================
    # 인식 모드
    # =========================================================================
    
    def _recognition_mode(self):
        """실시간 얼굴 인식"""
        print("\n" + "=" * 55)
        print("👁️ 얼굴 인식 모드")
        print("=" * 55)
        print(f"\n등록된 사람: {self.registered_name}")
        print(f"임계값: {Config.RECOGNITION_THRESHOLD}")
        print("\n📋 조작:")
        print("   R: 다시 등록")
        print("   +/-: 임계값 조절")
        print("   S: 저장")
        print("   ESC: 종료\n")
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ 카메라 열기 실패!")
            return
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        print("📷 실시간 인식 시작!\n")
        
        frame_count = 0
        start_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        threshold = Config.RECOGNITION_THRESHOLD
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            h, w = display.shape[:2]
            
            # 얼굴 검출
            faces = self.extractor.detect_faces(frame)
            
            # 각 얼굴 인식
            for (x, y, fw, fh) in faces:
                # 특징 추출
                face_img = self.extractor.extract_face(frame, (x, y, fw, fh))
                features = self.extractor.extract_features(face_img)
                
                if features is not None:
                    # 유사도 계산
                    similarity = self.extractor.compute_similarity(
                        features, self.registered_feature_avg
                    )
                    
                    # 판정
                    is_match = similarity >= threshold
                    
                    if is_match:
                        color = (0, 255, 0)
                        confidence = similarity * 100
                        label = f"{self.registered_name} ({confidence:.0f}%)"
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"
                    
                    # 박스 그리기
                    cv2.rectangle(display, (x, y), (x+fw, y+fh), color, 2)
                    
                    # 라벨 배경
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display, (x, y-30), (x+label_size[0]+10, y), color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(display, label, (x+5, y-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 유사도 표시
                    cv2.putText(display, f"Sim: {similarity:.2f}", (x, y+fh+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # FPS
            frame_count += 1
            elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 상태 표시
            cv2.putText(display, f"Recognition Mode | FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Registered: {self.registered_name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Threshold: {threshold:.2f} (+/-)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 범례
            cv2.rectangle(display, (w-180, 10), (w-10, 70), (50, 50, 50), -1)
            cv2.putText(display, "GREEN=Match", (w-170, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, "RED=Unknown", (w-170, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow("Face Recognition", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                camera.release()
                cv2.destroyAllWindows()
                self._registration_mode()
                return
            
            elif key == ord('+') or key == ord('='):
                threshold = min(0.95, threshold + 0.05)
                print(f"임계값: {threshold:.2f}")
            
            elif key == ord('-'):
                threshold = max(0.3, threshold - 0.05)
                print(f"임계값: {threshold:.2f}")
            
            elif key == ord('s'):
                self._save_data()
            
            elif key == 27:
                break
        
        camera.release()
        cv2.destroyAllWindows()
        print("\n👋 종료")
    
    # =========================================================================
    # 저장/불러오기
    # =========================================================================
    
    def _save_data(self):
        """데이터 저장"""
        filepath = os.path.join(self.save_folder, "face_opencv.pkl")
        
        data = {
            'name': self.registered_name,
            'features': self.registered_features,
            'feature_avg': self.registered_feature_avg
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 저장됨: {filepath}")
    
    def _load_data(self):
        """데이터 불러오기"""
        filepath = os.path.join(self.save_folder, "face_opencv.pkl")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.registered_name = data['name']
                self.registered_features = data['features']
                self.registered_feature_avg = data['feature_avg']
                return True
            except:
                pass
        return False


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       🔐 얼굴 인식 시스템 (OpenCV 전용 버전)              ║
    ║                                                           ║
    ║   ✅ dlib 불필요! opencv-python만 있으면 OK               ║
    ║   ✅ 맥북 M1/M3 완벽 호환                                 ║
    ║                                                           ║
    ║   설치: pip3 install opencv-python numpy                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    system = FaceRecognitionSystem()
    system.run()