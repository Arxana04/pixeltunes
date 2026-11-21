"""Generate public/covers/02_pokemon_pikachu.gif.

Hand-placed 60x60 pixel art scaled 4x to 240x240, 4 frames at 220 ms:
tail wag, a blink, swaying grass and twinkling sparkles.

    python3 scripts/generate_pikachu_cover.py   (needs Pillow)
"""
from pathlib import Path
from PIL import Image

GRID, SCALE = 60, 4
OUT = Path(__file__).resolve().parent.parent / "public/covers/02_pokemon_pikachu.gif"

C = {
    "O": (42, 27, 20),     # outline
    "Y": (248, 208, 56),   # yellow
    "S": (214, 158, 32),   # yellow shade
    "K": (24, 16, 26),     # ear tips / eyes
    "W": (255, 255, 255),  # eye glint
    "R": (232, 80, 58),    # cheeks
    "B": (138, 82, 40),    # tail base
    "P": (255, 92, 92),    # mouth inside
}
SKY_TOP, SKY_BOT = (150, 200, 238), (196, 226, 246)
MOUNT_FAR, MOUNT_NEAR = (128, 160, 206), (98, 132, 186)
GRASS_TOP, GRASS_BOT, GRASS_TUFT = (110, 178, 84), (78, 146, 66), (150, 206, 110)
BALL_RED, BALL_WHITE, BALL_INK = (226, 56, 56), (250, 250, 250), (36, 30, 64)

BODY = [
    ".OO..................OO.",
    ".OKO................OKO.",
    ".OKKO..............OKKO.",
    "..OKKO............OKKO..",
    "..OKYO............OYKO..",
    "...OYYO..........OYYO...",
    "...OYYO..........OYYO...",
    "....OYYO........OYYO....",
    "....OYYYOOOOOOOOOYYO....",
    "....OYYYYYYYYYYYYYYO....",
    "...OYYYYYYYYYYYYYYYYO...",
    "...OYYYYYYYYYYYYYYYYO...",
    "..OYYYKWYYYYYYYYKWYYYO..",
    "..OYYYKKYYYYYYYYKKYYYO..",
    "..ORRYYYYYYYYYYYYYYRRO..",
    "..ORRRYYYYOYYOYYYYRRRO..",
    "..ORRYYYYYYOOYYYYYYRRO..",
    "...OYYYYYYYYYYYYYYYYO...",
    "...OOYYYYYYYYYYYYYYOO...",
    ".....OYYYYYYYYYYYYO.....",
    "....OYYOYYYYYYYYOYYO....",
    "....OYYYOYYYYYYOYYYO....",
    "...OYYYYYYYYYYYYYYYYO...",
    "...OYYYYYYYYYYYYYYYYO...",
    "...OYYYYYYYYYYYYYYYYO...",
    "...OSYYYYYYYYYYYYYYSO...",
    "....OSSYYYYYYYYYYSSO....",
    "....OOSSSSOOOOSSSSOO....",
    "...OYYYYO......OYYYYO...",
    "...OOOOOO......OOOOOO...",
]
TAIL = [
    "....OOOOO.",
    "...OYYYYYO",
    "...OYYYYYO",
    "..OYYYYYO.",
    "..OYYYYO..",
    ".OOOYYO...",
    "OYYYYYO...",
    "OYYYYO....",
    ".OOYYO....",
    "..OYYO....",
    "..OBBOO...",
    "...OBBBO..",
    "....OBBBO.",
    ".....OOO..",
]
EYE_ROWS = (12, 13)


def check(sprite, name):
    widths = {len(r) for r in sprite}
    assert len(widths) == 1, f"{name} rows have uneven widths: {widths}"


def stamp(px, sprite, x0, y0):
    for y, row in enumerate(sprite):
        for x, ch in enumerate(row):
            if ch != ".":
                px[x0 + x, y0 + y] = C[ch]


def blink(sprite):
    rows = list(sprite)
    top, bot = EYE_ROWS
    rows[top] = rows[top].replace("K", "Y").replace("W", "Y")
    rows[bot] = rows[bot].replace("K", "O")
    return rows


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def background(px, frame):
    for y in range(GRID):
        for x in range(GRID):
            px[x, y] = lerp(SKY_TOP, SKY_BOT, y / 36) if y < 38 else (
                GRASS_TOP if y < 46 else GRASS_BOT)
    for peak_x, peak_y, half, color in ((16, 22, 18, MOUNT_FAR), (44, 20, 20, MOUNT_FAR),
                                         (30, 26, 16, MOUNT_NEAR)):
        for y in range(peak_y, 38):
            w = (y - peak_y) * half // (38 - peak_y)
            for x in range(max(0, peak_x - w), min(GRID, peak_x + w + 1)):
                px[x, y] = color
    sway = (0, 1, 0, -1)[frame]
    for tx in (6, 13, 43, 52):
        for dy in range(3):
            px[tx + (sway if dy == 0 else 0), 44 - dy] = GRASS_TUFT
    for i, (sx, sy) in enumerate(((8, 8), (50, 6), (41, 14), (12, 30))):
        if (i + frame) % 2 == 0:
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                px[sx + dx, sy + dy] = (255, 255, 230)
        else:
            px[sx, sy] = (255, 255, 230)


def pokeball(px, x0, y0):
    size = 10
    for y in range(size):
        for x in range(size):
            dx, dy = x - 4.5, y - 4.5
            d = dx * dx + dy * dy
            if d > 25:
                continue
            if d > 16 or y in (4, 5):
                px[x0 + x, y0 + y] = BALL_INK
            else:
                px[x0 + x, y0 + y] = BALL_RED if y < 4 else BALL_WHITE
    for x, y in ((4, 3), (5, 3), (3, 4), (6, 4), (3, 5), (6, 5), (4, 6), (5, 6)):
        px[x0 + x, y0 + y] = BALL_INK
    for x, y in ((4, 4), (5, 4), (4, 5), (5, 5)):
        px[x0 + x, y0 + y] = BALL_WHITE


def frame_image(frame):
    img = Image.new("RGB", (GRID, GRID))
    px = img.load()
    background(px, frame)
    pokeball(px, 46, 46)
    bx, by = 21, 25
    stamp(px, TAIL, bx - 6, by + 11 + (0, -1, 0, 1)[frame])
    stamp(px, blink(BODY) if frame == 3 else BODY, bx, by)
    return img.resize((GRID * SCALE, GRID * SCALE), Image.NEAREST)


def main():
    check(BODY, "BODY")
    check(TAIL, "TAIL")
    frames = [frame_image(i) for i in range(4)]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=220, loop=0,
                   optimize=False, disposal=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
