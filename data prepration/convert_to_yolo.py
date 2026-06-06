import os
import json
import shutil

ANTI_UAV_DIR = r"E:\Drone\Anti-UAV410"
OUTPUT_DIR = r"E:\Drone\irddt_dataset"
IMG_WIDTH = 640
IMG_HEIGHT = 512
FRAME_STEP = 5       # Taking every 5th frame
SPLITS = ["train", "val"]


def convert_bbox(x, y, w, h):
    # Converting [x, y, w, h] (top-left pixels) to YOLO [cx, cy, w, h] (normalized 0-1)
    cx = (x + w / 2) / IMG_WIDTH
    cy = (y + h / 2) / IMG_HEIGHT
    nw = w / IMG_WIDTH
    nh = h / IMG_HEIGHT

    # Clamping to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


def process_sequence(seq_dir, split, img_out, lbl_out, stats):
    # Process one video sequence: subsample frames, create YOLO labels.
    seq_name = os.path.basename(seq_dir)
    label_path = os.path.join(seq_dir, "IR_label.json")

    with open(label_path, "r") as f:
        data = json.load(f)

    exist_flags = data["exist"]
    gt_rects = data["gt_rect"]
    total_frames = len(exist_flags)

    for i in range(0, total_frames, FRAME_STEP):
        frame_num = i + 1  # Filenames are 1-indexed (000001.jpg)
        frame_file = f"{frame_num:06d}.jpg"
        src_img = os.path.join(seq_dir, frame_file)

        if not os.path.exists(src_img):
            continue

        # Unique output name: sequencename_framenum
        out_name = f"{seq_name}_{frame_num:06d}"
        dst_img = os.path.join(img_out, split, f"{out_name}.jpg")
        dst_lbl = os.path.join(lbl_out, split, f"{out_name}.txt")

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Create label file
        if exist_flags[i] == 1:
            x, y, w, h = gt_rects[i]
            if w > 0 and h > 0:
                cx, cy, nw, nh = convert_bbox(x, y, w, h)
                with open(dst_lbl, "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                stats["positive"] += 1
            else:
                # exist=1 but invalid bbox → negative sample
                open(dst_lbl, "w").close()
                stats["negative"] += 1
        else:
            # No drone in frame → empty label (negative sample)
            open(dst_lbl, "w").close()
            stats["negative"] += 1

        stats["total"] += 1


def main():
    # Creating output directories
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    img_out = os.path.join(OUTPUT_DIR, "images")
    lbl_out = os.path.join(OUTPUT_DIR, "labels")

    for split in SPLITS:
        print(f"Processing: {split}")
        split_dir = os.path.join(ANTI_UAV_DIR, split)
        sequences = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        stats = {"total": 0, "positive": 0, "negative": 0}

        for idx, seq in enumerate(sequences):
            seq_dir = os.path.join(split_dir, seq)
            print(f"  [{idx+1:3d}/{len(sequences)}] {seq}", end="")
            before = stats["total"]
            process_sequence(seq_dir, split, img_out, lbl_out, stats)
            after = stats["total"]
            print(f"  -> {after - before} frames")

        print(f"    Total frames:    {stats['total']}")
        print(f"    With drone (+):  {stats['positive']}")
        print(f"    No drone (-):    {stats['negative']}")

    print(f"Dataset saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
