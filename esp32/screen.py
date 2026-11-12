"""ShoppingKart cart screen: runs on the ESP32-S3 (MicroPython) with an SH1106 OLED.

The Mac sends one command per line over USB and this draws it:
    HI                          welcome screen
    NAV|up|4|Milk|Dairy|2/5     arrow (up/left/right/down), metres, next item, zone, progress
    PICK|Milk|2/5               arrived: pick it up and scan it
    ADD|+ Milk|248|3            cart changed: what changed, cart total, item count
    BAD                         scanned code is not a product
    PAID|SK-4821|428|5          order, total, item count
    MAP|<base64>|Dairy 2/5      128x64 1-bit picture (rows packed MSB first) with a caption on top
Pressing BOOT prints CHECKOUT back to the Mac.

Wiring: OLED GND-GND, VCC-3V3, SCL-GPIO9, SDA-GPIO8.
    mpremote mip install github:robert-hh/SH1106
    mpremote cp esp32/screen.py :main.py + reset
"""

import binascii
import sys
import time
from array import array

import framebuf
import machine
import neopixel
import sh1106

d = sh1106.SH1106_I2C(128, 64, machine.I2C(0, scl=machine.Pin(9), sda=machine.Pin(8)), addr=0x3C)
d.sleep(False)
led = neopixel.NeoPixel(machine.Pin(48), 1)

# arrow pointing up, around (0, 0); rotated for the other directions
ARROW = ((0, -22), (20, -2), (8, -2), (8, 22), (-8, 22), (-8, -2), (-20, -2))
TURN = {"up": lambda x, y: (x, y), "right": lambda x, y: (-y, x),
        "down": lambda x, y: (-x, -y), "left": lambda x, y: (y, -x)}


def arrow(way, cx=24, cy=32):
    d.poly(cx, cy, array("h", [v for p in ARROW for v in TURN[way](*p)]), 1, True)


def text(s, y, x=None):
    d.text(s, (128 - 8 * len(s)) // 2 if x is None else x, y, 1)


def big(s, y, x=None):
    """Text at double size (the built-in font is 8x8)."""
    small = framebuf.FrameBuffer(bytearray(8 * len(s)), 8 * len(s), 8, framebuf.MONO_HLSB)
    small.text(s, 0, 0, 1)
    x = (128 - 16 * len(s)) // 2 if x is None else x
    for j in range(8):
        for i in range(8 * len(s)):
            if small.pixel(i, j):
                d.fill_rect(x + 2 * i, y + 2 * j, 2, 2, 1)


def flash(rgb):
    led[0] = rgb
    led.write()
    time.sleep_ms(300)
    led[0] = (0, 0, 0)
    led.write()


def show(line):
    f = line.strip().split("|")
    d.fill(0)
    if f[0] == "HI":
        big("Shopping", 12)
        big("Kart", 32)
    elif f[0] == "NAV":
        _, way, metres, item, zone, progress = f
        arrow(way)
        text(item[:9], 4, 52)
        big(metres + "m", 18, 52)
        text(zone[:9], 40, 52)
        text(progress, 52, 52)
    elif f[0] == "PICK":
        _, item, progress = f
        text("PICK UP", 4)
        big(item, 20) if len(item) <= 8 else text(item[:16], 24)
        text("scan it  " + progress, 52)
    elif f[0] == "ADD":
        _, item, total, count = f
        text(item[:16], 4)
        big("Rs" + total, 22)
        text(count + " items", 50)
    elif f[0] == "MAP":
        _, picture, caption = f
        d.blit(framebuf.FrameBuffer(bytearray(binascii.a2b_base64(picture)), 128, 64, framebuf.MONO_HLSB), 0, 0)
        text(caption[:16], 0)
    elif f[0] == "BAD":
        text("Unknown item", 20)
        text("try again", 36)
    elif f[0] == "PAID":
        _, order, total, count = f
        big("PAID", 2)
        text("Rs " + total + "  " + count + " items", 28)
        text(order, 46)
    else:
        raise ValueError(line)
    d.show()
    if f[0] in ("ADD", "PAID"):
        flash((0, 40, 0))
    elif f[0] == "BAD":
        flash((40, 0, 0))


last_press = 0


def boot_pressed(pin):
    """BOOT button = Checkout. The Mac app listens for this line."""
    global last_press
    if time.ticks_diff(time.ticks_ms(), last_press) > 1000:  # debounce
        last_press = time.ticks_ms()
        print("CHECKOUT")


machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP).irq(boot_pressed, machine.Pin.IRQ_FALLING)

show("HI")
while True:
    line = sys.stdin.readline()
    try:
        show(line)
        print("ok")
    except Exception as e:  # a bad line from the Mac shouldn't kill the screen
        print("error", repr(e))
