"""Turn a session.cast into a real-time 1920x1080 MP4 with a light terminal look.

  .venv/bin/python experiment/tools/cast_to_video.py <session.cast> <out.mp4>
"""
import json, subprocess, sys
import pyte
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS, FPS, W, H = 120, 40, 10, 1920, 1080
SIZE, FONT = 22, '/System/Library/Fonts/Menlo.ttc'
FALLBACKS = ['/System/Library/Fonts/Apple Symbols.ttf', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf', '/System/Library/Fonts/LastResort.otf']
BG, FG = (255, 255, 255), (28, 28, 28)
NAMED = {'black': (0, 0, 0), 'red': (194, 54, 33), 'green': (37, 136, 60), 'brown': (160, 120, 0), 'blue': (40, 80, 200),
         'magenta': (170, 50, 170), 'cyan': (20, 130, 150), 'white': (200, 200, 200), 'brightblack': (120, 120, 120),
         'brightred': (220, 80, 60), 'brightgreen': (60, 160, 80), 'brightbrown': (190, 150, 20), 'brightblue': (70, 110, 230),
         'brightmagenta': (200, 80, 200), 'brightcyan': (40, 160, 180), 'brightwhite': (240, 240, 240)}
regular, bold = ImageFont.truetype(FONT, SIZE, index=0), ImageFont.truetype(FONT, SIZE, index=1)
CW, CH = regular.getlength('M'), H / (ROWS + 1)
X0, Y0 = (W - CW * COLS) / 2, CH / 2


def colour(name, default):
    if name == 'default':
        return default
    if name in NAMED:
        return NAMED[name]
    try:
        return tuple(int(name[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


SWAP = {'\u23fa': '\u25cf', '\u23f5': '\u25b8'}   # no macOS font draws these two Claude Code symbols
fallbacks = [ImageFont.truetype(f, SIZE) for f in FALLBACKS[:2]]
_pick = {}


def blank(font):
    return bytes(font.getmask('\U000F0000'))


def pick(ch, font):
    # Menlo has no glyph for some Claude Code symbols; use the first font that really draws the character.
    if ch.isascii():
        return font
    if ch not in _pick:
        _pick[ch] = next((f for f in [regular] + fallbacks if bytes(f.getmask(ch)) != blank(f)), regular)
    return font if _pick[ch] is regular else _pick[ch]


def draw_screen(draw, screen):
    # Whole frame in two passes, so one row's fill can never cut the tails off the row above.
    draw.rectangle([0, 0, W, H], fill=BG)
    cells = []
    for y in range(ROWS):
        top, row = Y0 + y * CH, screen.buffer[y]
        for x in range(COLS):
            c = row[x]
            fg, bg = colour(c.fg, FG), colour(c.bg, BG)
            if c.reverse:
                fg, bg = bg, fg
            left = X0 + x * CW
            if bg != BG:
                draw.rectangle([left, top, left + CW, top + CH], fill=bg)
            if c.data.strip():
                cells.append((left, top, SWAP.get(c.data, c.data), bold if c.bold else regular, fg))
    for left, top, ch, font, fg in cells:
        draw.text((left, top), ch, font=pick(ch[0], font), fill=fg)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    events = [json.loads(l) for l in open(src) if l.strip()]
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.Stream(screen)
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', dst], stdin=subprocess.PIPE)
    i, frames, last = 0, int(events[-1][0] * FPS) + FPS, b''
    for f in range(frames):
        t = f / FPS
        while i < len(events) and events[i][0] <= t:
            stream.feed(events[i][1]); i += 1
        if screen.dirty or not last:
            draw_screen(draw, screen)
            screen.dirty.clear()
            last = img.tobytes()
        ff.stdin.write(last)
    ff.stdin.close()
    ff.wait()
    print(dst, f'{frames / FPS:.1f}s', 'ffmpeg exit', ff.returncode)


main()
