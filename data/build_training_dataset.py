import os
import random
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

def build_dataset(output_dir="data/dataset"):
    print("[DatasetBuilder] Initializing dataset generator for YOLO26x...")
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

    fire_model = YOLO("weights/yolov26-fire.pt")
    
    samples_created = 0

    # 1. Extract from youtube_drone_fire.mp4.webm (Real aerial wildfire & smoke)
    video_path = "samples/youtube_drone_fire.mp4.webm"
    if os.path.exists(video_path):
        print(f"[DatasetBuilder] Extracting fire & smoke samples from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // 100)
        
        frame_idx = 0
        while cap.isOpened() and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Predict fire & smoke with yolov26-fire
            h, w = frame.shape[:2]
            res = fire_model.predict(frame, conf=0.25, imgsz=1280, device="cuda:0", verbose=False)
            
            labels = []
            if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                boxes = res[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.int().cpu().tolist()
                
                for b_i in range(len(boxes)):
                    c = clss[b_i]
                    # yolov26-fire classes: 0: fire, 1: other, 2: smoke
                    target_cls = None
                    if c == 0: # fire
                        target_cls = 0
                    elif c == 2: # smoke
                        target_cls = 1
                    
                    if target_cls is not None:
                        bx1, by1, bx2, by2 = xyxy[b_i]
                        # Filter out video logo / watermarks in corners
                        if by1 < 300 and bx1 < 700:
                            continue
                        if bx1 > w * 0.88 and by1 < 220:
                            continue
                        
                        cx = ((bx1 + bx2) / 2.0) / w
                        cy = ((by1 + by2) / 2.0) / h
                        bw = (bx2 - bx1) / w
                        bh = (by2 - by1) / h
                        labels.append(f"{target_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            if labels:
                split = "val" if samples_created % 6 == 0 else "train"
                img_name = f"fire_{samples_created:04d}.jpg"
                # Resize to 1280 for training
                scaled = cv2.resize(frame, (1280, 720))
                cv2.imwrite(f"{output_dir}/images/{split}/{img_name}", scaled)
                with open(f"{output_dir}/labels/{split}/{img_name.replace('.jpg', '.txt')}", "w") as f:
                    f.write("\n".join(labels) + "\n")
                samples_created += 1
            
            frame_idx += step
        cap.release()
        print(f"[DatasetBuilder] Real fire/smoke samples created: {samples_created}")

    # 2. Extract from synthetic_drone_mission.mp4 (Facade with windows and spreading fire)
    syn_path = "samples/synthetic_drone_mission.mp4"
    if os.path.exists(syn_path):
        print(f"[DatasetBuilder] Extracting window & facade fire samples from {syn_path}...")
        cap = cv2.VideoCapture(syn_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // 60)
        
        # Windows geometry
        cols = 6
        rows = 4
        win_w = 120
        win_h = 160
        gap_x = 80
        gap_y = 70
        start_x = 350
        start_y = 150
        base_windows = []
        for r in range(rows):
            for c in range(cols):
                wx = start_x + c * (win_w + gap_x)
                wy = start_y + r * (win_h + gap_y)
                base_windows.append((wx, wy, win_w, win_h))
        
        fire_win = base_windows[2 * cols + 2]
        fire_fx = fire_win[0] + fire_win[2] // 2
        fire_fy = fire_win[1] + fire_win[3] // 2
        
        fps = 30.0
        frame_idx = 0
        syn_count = 0
        while cap.isOpened() and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            h, w = frame.shape[:2]
            t = frame_idx / fps
            import math
            drone_dx = int(25 * math.sin(t * 0.8) + 10 * math.sin(t * 2.1))
            drone_dy = int(18 * math.cos(t * 0.6) + 8 * math.cos(t * 1.7))
            zoom = 1.0 + 0.05 * math.sin(t * 0.4)
            
            labels = []
            # Window annotations (class 2)
            for (bx, by, bw, bh) in base_windows:
                cur_x = int((bx + drone_dx) * zoom)
                cur_y = int((by + drone_dy) * zoom)
                cur_w = int(bw * zoom)
                cur_h = int(bh * zoom)
                
                cx = (cur_x + cur_w / 2.0) / w
                cy = (cur_y + cur_h / 2.0) / h
                norm_w = cur_w / w
                norm_h = cur_h / h
                if 0 < cx < 1 and 0 < cy < 1:
                    labels.append(f"2 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            # Fire annotation if active (class 0)
            if t > 1.5:
                fire_r = min(110.0, 20.0 + (t - 2.0) * 8.5) if t > 2.0 else 20.0
                cur_fx = int((fire_fx + drone_dx) * zoom)
                cur_fy = int((fire_fy + drone_dy) * zoom)
                cur_r = int(fire_r * zoom)
                
                fcx = cur_fx / w
                fcy = cur_fy / h
                fw_n = (cur_r * 2.2) / w
                fh_n = (cur_r * 2.2) / h
                labels.append(f"0 {fcx:.6f} {fcy:.6f} {fw_n:.6f} {fh_n:.6f}")
                
                # Smoke annotation (class 1)
                smk_cy = (cur_fy - cur_r * 1.3) / h
                smk_h = (cur_r * 2.5) / h
                smk_w = (cur_r * 3.0) / w
                if 0 < smk_cy < 1:
                    labels.append(f"1 {fcx:.6f} {smk_cy:.6f} {smk_w:.6f} {smk_h:.6f}")
            
            if labels:
                split = "val" if syn_count % 6 == 0 else "train"
                img_name = f"syn_facade_{syn_count:04d}.jpg"
                scaled = cv2.resize(frame, (1280, 720))
                cv2.imwrite(f"{output_dir}/images/{split}/{img_name}", scaled)
                with open(f"{output_dir}/labels/{split}/{img_name.replace('.jpg', '.txt')}", "w") as f:
                    f.write("\n".join(labels) + "\n")
                syn_count += 1
                samples_created += 1
            
            frame_idx += step
        cap.release()
        print(f"[DatasetBuilder] Synthetic facade samples created: {syn_count}")

    # 3. Create Augmented Photorealistic Urban Building Facade Samples with Windows & Fire
    print("[DatasetBuilder] Generating photorealistic urban building facade samples with window grids & localized fires...")
    urban_count = 0
    for u_idx in range(60):
        # Generate varied realistic building facade
        canvas_h, canvas_w = 720, 1280
        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Random facade color (concrete gray, brick red, beige, glass blue)
        wall_palette = [
            (140, 145, 150), # gray concrete
            (90, 95, 140),   # brick red/brown
            (160, 180, 190), # sandstone beige
            (180, 170, 160)  # modern dark stone
        ]
        wall_color = random.choice(wall_palette)
        img[:] = wall_color
        
        # Add texture / horizontal facade seams
        for sy in range(0, canvas_h, random.randint(40, 70)):
            cv2.line(img, (0, sy), (canvas_w, sy), (max(0, wall_color[0]-25), max(0, wall_color[1]-25), max(0, wall_color[2]-25)), 2)
            
        # Grid of windows
        cols = random.randint(5, 8)
        rows = random.randint(3, 5)
        win_w = random.randint(65, 110)
        win_h = random.randint(90, 140)
        gap_x = random.randint(40, 70)
        gap_y = random.randint(35, 60)
        start_x = random.randint(50, 120)
        start_y = random.randint(40, 90)
        
        u_labels = []
        win_rects = []
        for r in range(rows):
            for c in range(cols):
                wx = start_x + c * (win_w + gap_x)
                wy = start_y + r * (win_h + gap_y)
                if wx + win_w < canvas_w and wy + win_h < canvas_h:
                    win_rects.append((wx, wy, win_w, win_h))
                    
                    # Draw window frame & glass
                    glass_color = (random.randint(110, 170), random.randint(90, 140), random.randint(50, 90)) # deep glass reflections
                    cv2.rectangle(img, (wx-3, wy-3), (wx+win_w+3, wy+win_h+3), (35, 40, 45), -1) # dark frame
                    cv2.rectangle(img, (wx, wy), (wx+win_w, wy+win_h), glass_color, -1)
                    # Window crossbars / mullions
                    cv2.line(img, (wx + win_w//2, wy), (wx + win_w//2, wy + win_h), (50, 55, 60), 2)
                    cv2.line(img, (wx, wy + win_h//2), (wx + win_w, wy + win_h//2), (50, 55, 60), 2)
                    
                    # YOLO label (2: window)
                    wcx = (wx + win_w / 2.0) / canvas_w
                    wcy = (wy + win_h / 2.0) / canvas_h
                    ww_n = win_w / canvas_w
                    wh_n = win_h / canvas_h
                    u_labels.append(f"2 {wcx:.6f} {wcy:.6f} {ww_n:.6f} {wh_n:.6f}")

        # Introduce localized fire in 1 or 2 windows for a subset of samples
        if random.random() < 0.75 and win_rects:
            burn_win = random.choice(win_rects)
            bwx, bwy, bww, bwh = burn_win
            
            # Fire flame
            fcx = bwx + bww // 2
            fcy = bwy + bwh // 2
            fr = random.randint(int(bww * 0.4), int(bww * 0.9))
            
            for _ in range(6):
                ox = random.randint(-fr//3, fr//3)
                oy = random.randint(-fr//2, fr//4)
                r_c = random.randint(int(fr*0.4), fr)
                cv2.circle(img, (fcx + ox, fcy + oy), r_c, (0, random.randint(180, 255), 255), -1)
                cv2.circle(img, (fcx + ox, fcy + oy), int(r_c * 1.3), (0, random.randint(40, 100), 255), 2)
                
            # Fire label (0: fire)
            n_fcx = fcx / canvas_w
            n_fcy = fcy / canvas_h
            n_fw = (fr * 2.2) / canvas_w
            n_fh = (fr * 2.2) / canvas_h
            u_labels.append(f"0 {n_fcx:.6f} {n_fcy:.6f} {n_fw:.6f} {n_fh:.6f}")
            
            # Smoke plume
            smk_y = fcy - fr - random.randint(30, 80)
            overlay = img.copy()
            cv2.ellipse(overlay, (fcx, smk_y), (int(fr * 1.4), int(fr * 0.8)), 0, 0, 360, (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            
            n_scx = fcx / canvas_w
            n_scy = smk_y / canvas_h
            n_sw = (fr * 2.8) / canvas_w
            n_sh = (fr * 1.6) / canvas_h
            u_labels.append(f"1 {n_scx:.6f} {n_scy:.6f} {n_sw:.6f} {n_sh:.6f}")

        split = "val" if u_idx % 6 == 0 else "train"
        img_name = f"urban_facade_{u_idx:04d}.jpg"
        cv2.imwrite(f"{output_dir}/images/{split}/{img_name}", img)
        with open(f"{output_dir}/labels/{split}/{img_name.replace('.jpg', '.txt')}", "w") as f:
            f.write("\n".join(u_labels) + "\n")
        urban_count += 1
        samples_created += 1

    print(f"[DatasetBuilder] Photorealistic urban samples created: {urban_count}")

    # Write data.yaml
    data_yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "fire",
            1: "smoke",
            2: "window"
        }
    }
    
    yaml_file = f"{output_dir}/data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)
    
    print(f"[DatasetBuilder] Dataset successfully prepared at {output_dir}!")
    print(f"[DatasetBuilder] Total images: {samples_created} | Config: {yaml_file}")
    return yaml_file

if __name__ == "__main__":
    build_dataset()
