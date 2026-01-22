"""
Study Alretness Monitor
Real-time drowsiness detection using OpenCV and MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from collections import deque


class AlertnessMonitor:
    """Monitors user alertness during study sessions using eye tracking."""
    
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, ear_threshold=0.25, consecutive_frames=20):
        """
        Initialize the alertness monitor.
        
        Args:
            ear_threshold: EAR value below which eyes are considered closed
            consecutive_frames: Frames below threshold to trigger alert
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        


        self.drowsy_counter = 0
        self.total_frames = 0
        self.drowsy_frames = 0
        self.alerts_triggered = 0
        self.ear_history = deque(maxlen=30)
        self.start_time = None
        
    def calculate_ear(self, landmarks, eye_indices, frame_w, frame_h):
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            points.append((x, y))
        
        points = np.array(points)
        
        # vert distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        h = np.linalg.norm(points[0] - points[3])
        
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with EAR."""
        self.total_frames += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        ear = 0.0
        is_drowsy = False
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w, h)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            self.ear_history.append(ear)
            
            for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
                points = []
                for idx in eye_indices:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
            
            # Check for drowsiness
            if ear < self.ear_threshold:
                self.drowsy_counter += 1
                self.drowsy_frames += 1
                
                if self.drowsy_counter >= self.consecutive_frames:
                    is_drowsy = True
                    if self.drowsy_counter == self.consecutive_frames:
                        self.alerts_triggered += 1
            else:
                self.drowsy_counter = 0
        
        # overlay
        frame = self._draw_overlay(frame, ear, is_drowsy, face_detected)
        
        return frame, ear, is_drowsy
    
    def _draw_overlay(self, frame, ear, is_drowsy, face_detected):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        
        color = (0, 255, 0) if ear >= self.ear_threshold else (0, 0, 255)
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Threshold
        cv2.putText(frame, f"Threshold: {self.ear_threshold}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Drowsy counter progress bar
        progress = min(self.drowsy_counter / self.consecutive_frames, 1.0)
        bar_w = 150
        cv2.rectangle(frame, (180, 15), (180 + bar_w, 35), (60, 60, 60), -1)
        cv2.rectangle(frame, (180, 15), (180 + int(bar_w * progress), 35),
                      (0, 0, 255) if progress > 0.7 else (0, 165, 255), -1)
        cv2.putText(frame, f"{self.drowsy_counter}/{self.consecutive_frames}",
                    (180 + bar_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Alerts counter
        cv2.putText(frame, f"Alerts: {self.alerts_triggered}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        # Session time
        if self.start_time:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (w - 120, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if is_drowsy:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "DROWSINESS ALERT!", (w // 2 - 130, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # No face warning
        if not face_detected:
            cv2.putText(frame, "No face detected", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def generate_report(self):
        """Generate session summary report."""
        duration = time.time() - self.start_time if self.start_time else 0
        mins, secs = divmod(int(duration), 60)
        
        avg_ear = np.mean(self.ear_history) if self.ear_history else 0
        drowsy_pct = (self.drowsy_frames / max(1, self.total_frames)) * 100
        focus_score = max(0, 100 - drowsy_pct - (self.alerts_triggered * 5))
        
        report = f"""
================================================================================
                        STUDY SESSION REPORT
================================================================================

Session Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration:         {mins} minutes, {secs} seconds

--------------------------------------------------------------------------------
                          PERFORMANCE METRICS
--------------------------------------------------------------------------------

Total Frames Processed:    {self.total_frames:,}
Average EAR:               {avg_ear:.3f}
Drowsy Frames:             {self.drowsy_frames:,} ({drowsy_pct:.1f}%)
Alerts Triggered:          {self.alerts_triggered}

--------------------------------------------------------------------------------
                            FOCUS SCORE
--------------------------------------------------------------------------------

                              {focus_score:.0f}/100
                         {'★' * int(focus_score // 20)}{'☆' * (5 - int(focus_score // 20))}

"""
        if focus_score >= 80:
            report += "Excellent focus! Keep up the great work.\n"
        elif focus_score >= 60:
            report += "Good session. Consider taking short breaks to stay alert.\n"
        elif focus_score >= 40:
            report += "Moderate alertness. Try studying at a different time or get more rest.\n"
        else:
            report += "Low alertness detected. Please ensure adequate sleep before studying.\n"

        report += """
================================================================================
                              END OF REPORT
================================================================================
"""
        return report
    
    def run(self):
        """Run the alertness monitor."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.start_time = time.time()
        
        print("\n" + "=" * 50)
        print("STUDY ALERTNESS MONITOR")
        print("=" * 50)
        print("Press 'q' or ESC to quit and view report")
        print("=" * 50 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                frame, ear, is_drowsy = self.process_frame(frame)
                
                cv2.imshow("Study Alertness Monitor", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            # Generate and print report
            report = self.generate_report()
            print(report)
            # Save report to file
            report_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"Report saved to: {report_filename}")


def main():
    """Main entry point."""
    monitor = AlertnessMonitor(
        ear_threshold=0.25,
        consecutive_frames=20
    )
    monitor.run()


if __name__ == "__main__":
    main()
