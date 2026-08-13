from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SCALE = 4


def point(value: int) -> int:
    return value * SCALE


def box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(point(value) for value in values)


canvas = Image.new("RGBA", (point(512), point(512)), (0, 0, 0, 0))
draw = ImageDraw.Draw(canvas)
dark = "#171814"
lime = "#A8FF63"
cream = "#EEE9DE"

draw.rounded_rectangle(box((16, 16, 496, 496)), radius=point(112), fill=dark)
for start, end in [
    ((72, 146), (148, 146)), ((364, 146), (440, 146)),
    ((72, 256), (126, 256)), ((386, 256), (440, 256)),
    ((72, 366), (160, 366)), ((352, 366), (440, 366)),
]:
    draw.line(
        (point(start[0]), point(start[1]), point(end[0]), point(end[1])),
        fill=lime,
        width=point(34),
    )
    for x, y in (start, end):
        draw.ellipse(box((x - 17, y - 17, x + 17, y + 17)), fill=lime)

draw.rounded_rectangle(box((190, 74, 322, 318)), radius=point(66), fill=cream)
for y in (126, 169, 212):
    draw.line(box((221, y, 291, y)), fill=dark, width=point(18))

draw.line(box((150, 238, 150, 288)), fill=lime, width=point(34))
draw.line(box((362, 238, 362, 288)), fill=lime, width=point(34))
draw.arc(box((150, 182, 362, 394)), start=0, end=180, fill=lime, width=point(34))
draw.line(box((256, 394, 256, 444)), fill=cream, width=point(34))
draw.line(box((200, 444, 312, 444)), fill=cream, width=point(34))

icon = canvas.resize((512, 512), Image.Resampling.LANCZOS)
png_path = ASSETS / "pitchroom-icon.png"
ico_path = ASSETS / "pitchroom-icon.ico"
icon.save(png_path, optimize=True)
icon.save(
    ico_path,
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
print(png_path)
print(ico_path)
