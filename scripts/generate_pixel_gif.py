import os
import cv2
import numpy as np
from PIL import Image

def main():
    src_path = "/Users/mahipatni/.gemini/antigravity/brain/22864890-f343-4b6d-97ff-c0a3afc3a929/.user_uploaded/media_1794507756129_e4de156e.jpg"
    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {src_path}")
    
    h, w = img.shape[:2]
    print(f"Loaded image: {w}x{h}")
    
    # 1. Segment Daisies
    flower_configs = [
        {"center": (89, 652), "rad": 115, "phase": 0.0, "bob_amp": 6.5, "sway_amp": 3.0, "rot_amp": 1.8},
        {"center": (215, 808), "rad": 145, "phase": 2.1, "bob_amp": 7.5, "sway_amp": 3.5, "rot_amp": -2.2},
        {"center": (583, 402), "rad": 130, "phase": 4.2, "bob_amp": 7.0, "sway_amp": 3.2, "rot_amp": 2.0},
    ]
    
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    daisy_sprites = []
    total_flower_mask = np.zeros((h, w), dtype=np.uint8)
    
    for idx, cfg in enumerate(flower_configs):
        cy, cx = cfg["center"]
        rad = cfg["rad"]
        Y, X = np.ogrid[:h, :w]
        in_circle = (Y - cy)**2 + (X - cx)**2 <= rad**2
        
        is_center = (r > 150) & (g > 90) & (b < 120) & in_circle
        is_petal = in_circle & (V > 180) & ((S < 75) | ((H > 15) & (H < 45)))
        mask = (is_center | is_petal).astype(np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            lbl = labels[cy, cx]
            if lbl == 0:
                nearby = labels[max(0, cy-15):min(h, cy+15), max(0, cx-15):min(w, cx+15)]
                vals = nearby[nearby > 0]
                lbl = np.bincount(vals).argmax() if len(vals) > 0 else 1
            mask = (labels == lbl).astype(np.uint8)
            
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        total_flower_mask = np.maximum(total_flower_mask, mask)
        
        # Bounding box
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        
        # Sprite with alpha
        sprite = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        sprite[:, :, :3] = img[y0:y1, x0:x1]
        sprite[:, :, 3] = mask[y0:y1, x0:x1] * 255
        
        # Relative center
        rel_cy = float(cy - y0)
        rel_cx = float(cx - x0)
        
        daisy_sprites.append({
            "sprite": sprite,
            "y0": y0, "x0": x0,
            "rel_cy": rel_cy, "rel_cx": rel_cx,
            "config": cfg
        })
        print(f"Daisy {idx+1} extracted: shape {sprite.shape}, center ({rel_cy}, {rel_cx})")
        
    # Inpaint flowers from background
    print("Inpainting flowers from background...")
    dilated_flower_mask = cv2.dilate(total_flower_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    bg_clean = cv2.inpaint(img, dilated_flower_mask, 9, cv2.INPAINT_TELEA)
    
    # Inpaint static sparkles from background
    sparkle_mask = np.zeros((h, w), dtype=np.uint8)
    for cy, cx in [(250, 350), (880, 770), (565, 885)]:
        cv2.circle(sparkle_mask, (cx, cy), 22, 255, -1)
    bg_clean = cv2.inpaint(bg_clean, sparkle_mask, 7, cv2.INPAINT_TELEA)
    
    # Extract Caustics mask for subtle shimmer
    bg_hsv = cv2.cvtColor(bg_clean, cv2.COLOR_BGR2HSV)
    bg_v = bg_hsv[:,:,2]
    bg_s = bg_hsv[:,:,1]
    is_caustic = ((bg_v > 210) & (bg_s < 100)).astype(np.float32)
    is_caustic = cv2.GaussianBlur(is_caustic, (7, 7), 2)
    
    # Grid coordinates
    Y, X = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    
    num_frames = 32
    frames_raw = []
    
    # Sparkle definitions: (y, x, frame_offset)
    sparkle_stars = [
        (250, 350, 0),     # Star A: top-left (original)
        (880, 770, 8),     # Star B: bottom-right (original)
        (565, 885, 16),    # Star C: middle-right (original)
        (160, 345, 24),    # Star D: top-left small dot
        (393, 654, 4),     # Star E: mid-right caustics
        (720, 210, 12),    # Star F: lower-left caustics
        (85, 890, 20),     # Star G: upper-right caustics
    ]
    
    print(f"Generating {num_frames} animation frames...")
    for frame_idx in range(num_frames):
        theta = 2.0 * np.pi * frame_idx / num_frames
        
        # 1. Harmonic water wave displacement
        # Smooth circular orbital wave components
        disp_x1 = 6.0 * np.sin(theta + 2.0 * np.pi * Y / 340.0 + 2.0 * np.pi * X / 480.0)
        disp_y1 = 5.0 * np.cos(theta + 2.0 * np.pi * X / 340.0 - 2.0 * np.pi * Y / 420.0)
        
        # Secondary cross-flow ripple
        disp_x2 = 2.5 * np.cos(2.0 * theta + 2.0 * np.pi * (X - Y) / 240.0)
        disp_y2 = 2.2 * np.sin(2.0 * theta + 2.0 * np.pi * (X + Y) / 240.0)
        
        # Total displacement map
        map_x = np.clip(X + disp_x1 + disp_x2, 0, w - 1).astype(np.float32)
        map_y = np.clip(Y + disp_y1 + disp_y2, 0, h - 1).astype(np.float32)
        
        # Remap water
        frame_water = cv2.remap(bg_clean, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Subtle traveling caustic shimmer (glistening sunlight)
        caustic_wave = np.sin(theta + (X + Y) / 140.0)
        shimmer_intensity = (is_caustic * caustic_wave * 10.0)[:, :, None]
        frame_water = np.clip(frame_water.astype(np.float32) + shimmer_intensity, 0, 255).astype(np.uint8)
        
        # 2. Render Bobbing & Floating Daisies
        frame_comp = frame_water.copy()
        
        for d in daisy_sprites:
            cfg = d["config"]
            p = cfg["phase"]
            
            # Buoyancy physics
            dy = cfg["bob_amp"] * np.sin(theta + p)
            dx = cfg["sway_amp"] * np.cos(theta + p)
            rot = cfg["rot_amp"] * np.sin(2.0 * theta + p + 0.4)
            
            sprite = d["sprite"]
            sh, sw = sprite.shape[:2]
            
            # Center of rotation (floats for cv2)
            center_x = float(d["rel_cx"])
            center_y = float(d["rel_cy"])
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), float(rot), 1.0)
            M[0, 2] += float(dx)
            M[1, 2] += float(dy)
            
            # Target bounding position
            ty0 = int(round(d["y0"]))
            tx0 = int(round(d["x0"]))
            
            # Subtle water shadow beneath daisy (floating contact shadow)
            shadow_mask = cv2.warpAffine(sprite[:, :, 3], M, (sw, sh), flags=cv2.INTER_LINEAR)
            shadow_dilated = cv2.dilate(shadow_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            shadow_fringe = np.clip(shadow_dilated.astype(float) - shadow_mask.astype(float), 0, 255) / 255.0
            
            s_offset_y = 6
            s_offset_x = 2
            sy_start = max(0, ty0 + s_offset_y)
            sy_end = min(h, ty0 + s_offset_y + sh)
            sx_start = max(0, tx0 + s_offset_x)
            sx_end = min(w, tx0 + s_offset_x + sw)
            
            fry0 = sy_start - (ty0 + s_offset_y)
            fry1 = fry0 + (sy_end - sy_start)
            frx0 = sx_start - (tx0 + s_offset_x)
            frx1 = frx0 + (sx_end - sx_start)
            
            if sy_end > sy_start and sx_end > sx_start:
                fringe_crop = shadow_fringe[fry0:fry1, frx0:frx1, None]
                # Darken water contact shadow
                frame_comp[sy_start:sy_end, sx_start:sx_end] = np.clip(
                    frame_comp[sy_start:sy_end, sx_start:sx_end].astype(float) * (1.0 - fringe_crop * 0.25),
                    0, 255
                ).astype(np.uint8)
            
            # Warp sprite
            warped_sprite = cv2.warpAffine(sprite, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            
            # Composite daisy onto water
            cy0 = max(0, ty0)
            cy1 = min(h, ty0 + sh)
            cx0 = max(0, tx0)
            cx1 = min(w, tx0 + sw)
            
            wy0 = cy0 - ty0
            wy1 = wy0 + (cy1 - cy0)
            wx0 = cx0 - tx0
            wx1 = wx0 + (cx1 - cx0)
            
            if cy1 > cy0 and cx1 > cx0:
                alpha = warped_sprite[wy0:wy1, wx0:wx1, 3:4].astype(float) / 255.0
                fg_rgb = warped_sprite[wy0:wy1, wx0:wx1, :3].astype(float)
                bg_rgb = frame_comp[cy0:cy1, cx0:cx1].astype(float)
                
                frame_comp[cy0:cy1, cx0:cx1] = np.clip(bg_rgb * (1.0 - alpha) + fg_rgb * alpha, 0, 255).astype(np.uint8)
                
        # 3. Render Retro Pixel Sparkles
        for sy, sx, offset in sparkle_stars:
            sparkle_frame = (frame_idx - offset) % num_frames
            if sparkle_frame < 8:
                draw_pixel_block_sparkle(frame_comp, sy, sx, sparkle_frame, block_size=9)
                
        frames_raw.append(frame_comp)
        
    print("Downscaling and optimizing frames...")
    frames_512 = []
    frames_240 = []
    
    for f in frames_raw:
        # 512x512
        f512 = cv2.resize(f, (512, 512), interpolation=cv2.INTER_AREA)
        f512_rgb = cv2.cvtColor(f512, cv2.COLOR_BGR2RGB)
        frames_512.append(Image.fromarray(f512_rgb))
        
        # 240x240
        f240 = cv2.resize(f, (240, 240), interpolation=cv2.INTER_AREA)
        f240_rgb = cv2.cvtColor(f240, cv2.COLOR_BGR2RGB)
        frames_240.append(Image.fromarray(f240_rgb))
        
    print("Saving 512x512 GIFs...")
    out_512_assets = "assets/pixel-daisy-pond.gif"
    out_512_chat = "/Users/mahipatni/.gemini/antigravity/brain/22864890-f343-4b6d-97ff-c0a3afc3a929/pixel-daisy-pond.gif"
    save_quantized_gif(frames_512, out_512_assets, colors=128, duration=90)
    save_quantized_gif(frames_512, out_512_chat, duration=90)
    print(f"Saved {out_512_assets} ({os.path.getsize(out_512_assets)} bytes)")
    
    print("Saving 240x240 cover GIF...")
    out_240_cover = "public/covers/24_daisy_pond.gif"
    save_quantized_gif(frames_240, out_240_cover, colors=96, duration=90)
    print(f"Saved {out_240_cover} ({os.path.getsize(out_240_cover)} bytes)")

def draw_pixel_block_sparkle(img, cy, cx, stage, block_size=9):
    """
    Renders pixel-perfect retro blocky 4-point diamond sparkle.
    cy, cx: center in 1024 canvas
    stage: 0 to 7
    """
    h, w = img.shape[:2]
    B = block_size
    half = B // 2
    
    center_y = (cy // B) * B + half
    center_x = (cx // B) * B + half
    
    C_CORE = (255, 255, 255)
    C_MID = (245, 255, 255)
    C_OUTER = (210, 245, 252)
    
    def fill_block(by, bx, color):
        y0 = max(0, by - half)
        y1 = min(h, by + half + 1)
        x0 = max(0, bx - half)
        x1 = min(w, bx + half + 1)
        if y1 > y0 and x1 > x0:
            img[y0:y1, x0:x1] = color

    if stage in (0, 7):
        fill_block(center_y, center_x, C_MID)
    elif stage in (1, 6):
        fill_block(center_y, center_x, C_CORE)
        for dy, dx in [(-B, 0), (B, 0), (0, -B), (0, B)]:
            fill_block(center_y + dy, center_x + dx, C_OUTER)
    elif stage in (2, 5):
        fill_block(center_y, center_x, C_CORE)
        for dy, dx in [(-B, 0), (B, 0), (0, -B), (0, B)]:
            fill_block(center_y + dy, center_x + dx, C_MID)
        for dy, dx in [(-2*B, 0), (2*B, 0), (0, -2*B), (0, 2*B), (-B, -B), (-B, B), (B, -B), (B, B)]:
            fill_block(center_y + dy, center_x + dx, C_OUTER)
    elif stage in (3, 4):
        fill_block(center_y, center_x, C_CORE)
        for dy, dx in [(-B, 0), (B, 0), (0, -B), (0, B)]:
            fill_block(center_y + dy, center_x + dx, C_CORE)
        for dy, dx in [(-B, -B), (-B, B), (B, -B), (B, B)]:
            fill_block(center_y + dy, center_x + dx, C_MID)
        for dy, dx in [(-2*B, 0), (2*B, 0), (0, -2*B), (0, 2*B)]:
            fill_block(center_y + dy, center_x + dx, C_MID)
        for dy, dx in [(-3*B, 0), (3*B, 0), (0, -3*B), (0, 3*B)]:
            fill_block(center_y + dy, center_x + dx, C_OUTER)

def save_quantized_gif(frames, out_path, colors=128, duration=90):
    sample_frames = frames[::4]
    mw, mh = sample_frames[0].size
    montage = Image.new('RGB', (mw * 2, mh * (len(sample_frames) + 1) // 2))
    for i, sf in enumerate(sample_frames):
        mx = (i % 2) * mw
        my = (i // 2) * mh
        montage.paste(sf, (mx, my))
        
    master_palette_img = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    
    quantized_frames = []
    for f in frames:
        q = f.quantize(palette=master_palette_img, dither=Image.Dither.NONE)
        quantized_frames.append(q)
        
    quantized_frames[0].save(
        out_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )

if __name__ == "__main__":
    main()
