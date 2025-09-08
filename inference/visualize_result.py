#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image, ImageDraw

def make_overlay_slide(
    slide_id: str, tiles, is_meta,
    tile_size=512, out_dir=Path("./wsi_image_virchow2"), max_width=2000
):
    """
    추론된 타일 판정 결과를 기반으로 META/NON-META 오버레이 썸네일 생성
    - META: 빨간 반투명
    - NON-META: 검정 반투명
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    coords = []
    for tp in tiles:
        parts = tp.stem.split("_")
        try:
            # 뒤에서 3번째 = x, 2번째 = y (마지막은 배율, 무시)
            x, y = int(parts[-3]), int(parts[-2])
            coords.append((x, y))
        except Exception:
            print(f"⚠️ 좌표 파싱 실패: {tp.name}, (0,0)으로 처리")
            coords.append((0, 0))

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    W, H = max(xs) + tile_size, max(ys) + tile_size

    slide_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(slide_img, "RGBA")

    for tp, (x, y), meta in zip(tiles, coords, is_meta):
        try:
            tile = Image.open(tp).convert("RGB")
            slide_img.paste(tile, (x, y))
        except Exception as e:
            print(f"⚠️ 타일 로드 실패: {tp.name}, {e}")
            continue

        rect = [x, y, x+tile_size, y+tile_size]
        if meta:
            draw.rectangle(rect, fill=(255, 0, 0, 100))  # META = 빨간 반투명
        else:
            draw.rectangle(rect, fill=(0, 0, 0, 120))    # NON-META = 검정 반투명

    # 축소 (가로 max_width 기준)
    if W > max_width:
        ratio = max_width / float(W)
        new_size = (max_width, int(H * ratio))
        slide_img = slide_img.resize(new_size, Image.LANCZOS)

    out_path = out_dir / f"{slide_id}_overlay_thumb.jpg"
    slide_img.save(out_path, "JPEG", quality=90)
    print(f"  ✅ Overlay 썸네일 저장: {out_path}")
