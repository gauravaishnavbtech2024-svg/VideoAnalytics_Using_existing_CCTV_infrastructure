"""
FaceTrace Asset Generator
Generates realistic face crops and avatar assets for all reference cases and candidate frames.
"""
import os
import math
from PIL import Image, ImageDraw, ImageFilter

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "faces")
os.makedirs(ASSETS_DIR, exist_ok=True)

def create_face_crop(
    filename,
    skin_tone="#d4a373",
    hair_color="#1e1815",
    shirt_color="#cbd5e1",
    yaw=0,
    masked=False,
    dark=False,
    blurry=False,
    size=(320, 320)
):
    img = Image.new("RGBA", size, (248, 250, 252, 255))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Background subtle gradient / grid
    for y in range(0, h, 20):
        draw.line([(0, y), (w, y)], fill=(241, 245, 249, 180), width=1)
    for x in range(0, w, 20):
        draw.line([(x, 0), (x, h)], fill=(241, 245, 249, 180), width=1)

    # Torso / Shoulders
    torso_top = int(h * 0.72)
    draw.ellipse(
        [int(w * 0.15), torso_top, int(w * 0.85), int(h * 1.3)],
        fill=shirt_color
    )

    # Neck
    neck_w = int(w * 0.22)
    neck_left = int(w * 0.5 - neck_w / 2)
    neck_top = int(h * 0.52)
    draw.rectangle(
        [neck_left, neck_top, neck_left + neck_w, int(h * 0.74)],
        fill=skin_tone
    )

    # Face Shape (Oval)
    eye_shift = (yaw / 30.0) * (w * 0.08)
    head_cx = int(w * 0.5 + eye_shift * 0.3)
    head_cy = int(h * 0.44)
    head_rx = int(w * 0.24)
    head_ry = int(h * 0.30)

    draw.ellipse(
        [head_cx - head_rx, head_cy - head_ry, head_cx + head_rx, head_cy + head_ry],
        fill=skin_tone
    )

    # Hair
    hair_top = head_cy - head_ry - int(h * 0.03)
    hair_bottom = head_cy - int(head_ry * 0.15)
    draw.chord(
        [head_cx - head_rx - 4, hair_top, head_cx + head_rx + 4, hair_bottom + 30],
        start=170,
        end=370,
        fill=hair_color
    )

    # Eyebrows
    eb_y = head_cy - int(head_ry * 0.32)
    draw.line([head_cx - 45 + eye_shift, eb_y, head_cx - 15 + eye_shift, eb_y - 3], fill=hair_color, width=4)
    draw.line([head_cx + 15 + eye_shift, eb_y - 3, head_cx + 45 + eye_shift, eb_y], fill=hair_color, width=4)

    # Eyes
    eye_y = head_cy - int(head_ry * 0.15)
    # Left eye
    draw.ellipse([head_cx - 40 + eye_shift, eye_y - 6, head_cx - 24 + eye_shift, eye_y + 6], fill="#ffffff")
    draw.ellipse([head_cx - 35 + eye_shift, eye_y - 5, head_cx - 27 + eye_shift, eye_y + 5], fill="#1e293b")
    # Right eye
    draw.ellipse([head_cx + 24 + eye_shift, eye_y - 6, head_cx + 40 + eye_shift, eye_y + 6], fill="#ffffff")
    draw.ellipse([head_cx + 27 + eye_shift, eye_y - 5, head_cx + 35 + eye_shift, eye_y + 5], fill="#1e293b")

    # Nose
    nose_x = head_cx + eye_shift * 0.6
    nose_y = head_cy + int(head_ry * 0.12)
    draw.line([nose_x, nose_y - 14, nose_x - 3, nose_y + 4], fill="#b45309", width=2)
    draw.line([nose_x - 3, nose_y + 4, nose_x + 5, nose_y + 4], fill="#b45309", width=2)

    # Mouth
    mouth_y = head_cy + int(head_ry * 0.44)
    draw.arc(
        [head_cx - 22 + eye_shift * 0.5, mouth_y - 8, head_cx + 22 + eye_shift * 0.5, mouth_y + 8],
        start=20,
        end=160,
        fill="#9a3412",
        width=3
    )

    # Mask if occluded
    if masked:
        mask_top = head_cy + int(head_ry * 0.05)
        mask_bottom = head_cy + head_ry + 4
        mask_pts = [
            (head_cx - head_rx + 4, mask_top),
            (head_cx + head_rx - 4, mask_top),
            (head_cx + head_rx - 8, mask_bottom),
            (head_cx - head_rx + 8, mask_bottom),
        ]
        draw.polygon(mask_pts, fill="#475569")
        # Mask folds
        draw.line([head_cx - head_rx + 10, mask_top + 18, head_cx + head_rx - 10, mask_top + 18], fill="#334155", width=2)
        draw.line([head_cx - head_rx + 10, mask_top + 36, head_cx + head_rx - 10, mask_top + 36], fill="#334155", width=2)

    # Low light / Dark filter
    if dark:
        dark_layer = Image.new("RGBA", size, (10, 15, 26, 120))
        img = Image.alpha_composite(img, dark_layer)

    # Blurry / Low resolution
    if blurry:
        img = img.resize((w // 4, h // 4), Image.Resampling.BILINEAR)
        img = img.resize(size, Image.Resampling.NEAREST)

    target_path = os.path.join(ASSETS_DIR, filename)
    img.save(target_path, "PNG")
    print(f"Generated: {target_path}")

def generate_all():
    print("Generating FaceTrace assets in:", ASSETS_DIR)

    # Case 49017 (Selected main case)
    create_face_crop("txn_49017_best.png", skin_tone="#d4a373", hair_color="#181311", shirt_color="#cbd5e1", yaw=0)
    create_face_crop("wl_0012_record.png", skin_tone="#d4a373", hair_color="#181311", shirt_color="#bfdbfe", yaw=0)

    # 8 candidate frames for TXN-49017
    create_face_crop("f1_score75.png", skin_tone="#d4a373", yaw=-8, dark=True)
    create_face_crop("f2_score74.png", skin_tone="#d4a373", yaw=12)
    create_face_crop("f3_score70.png", skin_tone="#d4a373", masked=True)
    create_face_crop("f4_score75.png", skin_tone="#d4a373", masked=True)
    create_face_crop("f5_score74.png", skin_tone="#d4a373", yaw=-14)
    create_face_crop("f6_score71.png", skin_tone="#d4a373", yaw=18)
    create_face_crop("f7_score68.png", skin_tone="#d4a373", masked=True)
    create_face_crop("f8_score92_best.png", skin_tone="#d4a373", yaw=0)

    # Other cases in left panel
    create_face_crop("txn_48213.png", skin_tone="#c68642", hair_color="#1e1815", dark=True)
    create_face_crop("txn_48377.png", skin_tone="#c68642", hair_color="#1e1815", masked=True)
    create_face_crop("txn_48502.png", skin_tone="#f5d0b5", hair_color="#2b2320", blurry=True)
    create_face_crop("txn_48611.png", skin_tone="#b07038", hair_color="#141110", yaw=24)
    create_face_crop("txn_48790.png", skin_tone="#a06030", hair_color="#0d0b0a", dark=True, yaw=18)
    create_face_crop("txn_48934.png", skin_tone="#edd2b9", hair_color="#231d1b", yaw=0)

if __name__ == "__main__":
    generate_all()
