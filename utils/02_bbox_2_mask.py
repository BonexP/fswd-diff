from PIL import Image, ImageDraw

def bbox_to_mask(image, bbox, expand_ratio=0.2):
    w, h = image.size
    mask = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(mask)

    x1, y1, x2, y2 = bbox

    # 扩展bbox（关键）
    dw = (x2 - x1) * expand_ratio
    dh = (y2 - y1) * expand_ratio

    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw)
    y2 = min(h, y2 + dh)

    draw.rectangle([x1, y1, x2, y2], fill="white")

    return mask
