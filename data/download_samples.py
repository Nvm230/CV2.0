import os
import cv2
import numpy as np
import math
import random

def generate_synthetic_drone_mission(output_path: str='samples/synthetic_drone_mission.mp4', width: int=3840, height: int=2160, fps: int=30, duration_sec: int=15):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = fps * duration_sec
    print(f'[Samples] Generating {duration_sec}s ({total_frames} frames) realistic drone test video at {output_path}...')
    cols = 6
    rows = 4
    win_w = 120
    win_h = 160
    gap_x = 80
    gap_y = 70
    start_x = 350
    start_y = 150
    windows = []
    for r in range(rows):
        for c in range(cols):
            wx = start_x + c * (win_w + gap_x)
            wy = start_y + r * (win_h + gap_y)
            windows.append({'x': wx, 'y': wy, 'w': win_w, 'h': win_h, 'row': r, 'col': c})
    fire_origin_win = windows[2 * cols + 2]
    fire_fx = fire_origin_win['x'] + fire_origin_win['w'] // 2
    fire_fy = fire_origin_win['y'] + fire_origin_win['h'] // 2
    fire_radius = 20.0
    for i in range(total_frames):
        t = i / float(fps)
        drone_dx = int(25 * math.sin(t * 0.8) + 10 * math.sin(t * 2.1))
        drone_dy = int(18 * math.cos(t * 0.6) + 8 * math.cos(t * 1.7))
        zoom_factor = 1.0 + 0.05 * math.sin(t * 0.4)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (70, 75, 85)
        for py in range(0, height, 100):
            cv2.line(frame, (0, py), (width, py), (60, 65, 75), 1)
        if t > 2.0:
            fire_radius = min(110.0, 20.0 + (t - 2.0) * 8.5)
        for idx, win in enumerate(windows):
            wx = int((win['x'] + drone_dx) * zoom_factor)
            wy = int((win['y'] + drone_dy) * zoom_factor)
            ww = int(win['w'] * zoom_factor)
            wh = int(win['h'] * zoom_factor)
            cv2.rectangle(frame, (wx - 4, wy - 4), (wx + ww + 4, wy + wh + 4), (30, 35, 40), -1)
            cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (140, 110, 80), -1)
            cv2.line(frame, (wx + ww // 2, wy), (wx + ww // 2, wy + wh), (45, 50, 55), 3)
            cv2.line(frame, (wx, wy + wh // 2), (wx + ww, wy + wh // 2), (45, 50, 55), 3)
        if t > 1.5:
            cur_fx = int((fire_fx + drone_dx) * zoom_factor)
            cur_fy = int((fire_fy + drone_dy) * zoom_factor)
            cur_r = int(fire_radius * zoom_factor)
            for _ in range(8):
                ox = random.randint(-cur_r // 3, cur_r // 3)
                oy = random.randint(-cur_r // 2, cur_r // 4)
                r_flame = random.randint(int(cur_r * 0.4), int(cur_r * 0.9))
                cv2.circle(frame, (cur_fx + ox, cur_fy + oy), r_flame, (0, 215, 255), -1)
                cv2.circle(frame, (cur_fx + ox, cur_fy + oy), int(r_flame * 1.3), (0, 69, 255), 2)
            smoke_y = cur_fy - cur_r - int(t * 20 % 150)
            for sy in range(cur_fy - cur_r, max(0, cur_fy - 250), -30):
                smoke_w = int(cur_r * 1.2 + abs(cur_fy - sy) * 0.4)
                smoke_center_x = cur_fx + int(20 * math.sin(sy * 0.05 + t * 3))
                overlay = frame.copy()
                cv2.ellipse(overlay, (smoke_center_x, sy), (smoke_w, 20), 0, 0, 360, (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        out.write(frame)
    out.release()
    print(f'[Samples] Synthetic test video generated successfully: {output_path}')

def download_or_generate_samples():
    samples_dir = 'samples'
    os.makedirs(samples_dir, exist_ok=True)
    mission_video = os.path.join(samples_dir, 'synthetic_drone_mission.mp4')
    if not os.path.exists(mission_video):
        generate_synthetic_drone_mission(mission_video)
    cap = cv2.VideoCapture(mission_video)
    ret, frame = cap.read()
    if ret:
        ref_img_path = os.path.join(samples_dir, 'drone_building_test.jpg')
        cv2.imwrite(ref_img_path, frame)
        print(f'[Samples] Saved test image: {ref_img_path}')
    cap.release()
if __name__ == '__main__':
    download_or_generate_samples()
