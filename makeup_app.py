import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from facekit.makeup import BUILTIN_LOOKS, HAIR_LIFT, KEYS, LAYERS, PRESETS, STYLES, MakeupSession, bgr_to_hex  # noqa: E402
from facekit.viz import draw_box, draw_points  # noqa: E402

PANEL_W = 560
MAX_W = 1400
GAP = 4
MIN_H = 532   # the camera view is scaled to the panel height but never below this, so the window doesn't jump

ACCENT = (180, 105, 255)
DIM = (150, 150, 150)
TEXT = (30, 30, 30)
BG = (245, 245, 245)

SV_W = 256
SV_H = 110
HUE_H = 16

MASK_COLOURS = {
    "lips": (60, 40, 220), "blush": (150, 120, 240), "shadow": (200, 120, 200), "liner": (0, 0, 0),
    "lashes": (40, 40, 40), "skin": (120, 200, 240), "lens": (200, 160, 60), "brows": (40, 90, 140),
    "teeth": (255, 255, 255), "hair": (60, 160, 60), "mustache": (160, 60, 60), "beard": (120, 40, 90),
}
LAYER_KEYS = {"mustache": "T", "beard": "J"}  # layers that don't have a number key

ACTION_KEYS = {"l": "next", "r": "reset", "b": "split", "v": "side", "m": "masks", "p": "dots", "x": "box", "k": "label",
               "g": "unet", "s": "save_image"}


def make_hue_bar():
    hue = np.linspace(0, 179, SV_W).astype(np.uint8)
    h = np.tile(hue, (HUE_H, 1))
    s = np.full((HUE_H, SV_W), 255, np.uint8)
    v = np.full((HUE_H, SV_W), 255, np.uint8)
    return cv2.cvtColor(np.stack([h, s, v], -1), cv2.COLOR_HSV2BGR)


def make_sv_square(hue):
    """Saturation goes left to right, brightness goes top to bottom."""
    s = np.tile(np.linspace(0, 255, SV_W), (SV_H, 1))
    v = np.tile(np.linspace(255, 0, SV_H)[:, None], (1, SV_W))
    hsv = np.stack([np.full((SV_H, SV_W), hue), s, v], -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


HUE_BAR = make_hue_bar()


def slider_value(x, x0, x1):
    """Mouse x on a slider -> value 0..1."""
    x = min(max(x, x0), x1)
    return (x - x0) / (x1 - x0)


class ControlsPanel:
    """Draws the controls and turns clicks / keys into MakeupSession calls. No window code in here."""

    def __init__(self, session, save_dir):
        self.session = session
        self.save_dir = Path(save_dir)
        self.hits = []  # clickable areas: (x0, y0, x1, y1, action, arg)
        self.split = False
        self.side = False
        self.masks = False
        self.show_box = False
        self.show_dots = False
        self.show_label = False
        if session.warnings:
            self.status = "; ".join(session.warnings)
        else:
            self.status = "Click a layer, then a colour, style or strength."
        self.prompt = None
        self.typed = ""
        self.save_request = False
        self.quit = False
        self.drag_act = None
        self.picker = False
        self.hsv = (0, 0, 0)
        self.sv_cache = (None, None)  # (hue, square) so I don't redraw the square every frame


    def put(self, img, text, x, y, size=0.5, colour=TEXT):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)

    def button(self, img, x, y, w, h, label, action, arg=None, active=False, dim=False, right=0):
        if active:
            fill = ACCENT
            text_colour = (255, 255, 255)
        else:
            fill = (225, 225, 225)
            text_colour = DIM if dim else TEXT
        cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
        if len(label) < 12:
            size = 0.45
        else:
            size = 0.38
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        text_x = x + max(4, (w - right - text_w) // 2)
        text_y = y + (h + text_h) // 2
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, size, text_colour, 1, cv2.LINE_AA)
        self.hits.append((x, y, x + w, y + h, action, arg))

    def slider(self, img, y, label, value, action):
        self.put(img, label, 12, y + 16)
        x0 = 20
        x1 = PANEL_W - 20
        bar_y = y + 30
        knob_x = int(x0 + value * (x1 - x0))
        cv2.rectangle(img, (x0, bar_y), (x1, bar_y + 10), (210, 210, 210), -1)
        cv2.rectangle(img, (x0, bar_y), (knob_x, bar_y + 10), ACCENT, -1)
        cv2.circle(img, (knob_x, bar_y + 5), 9, (90, 60, 160), -1)
        self.hits.append((x0 - 10, bar_y - 10, x1 + 10, bar_y + 20, action, (x0, x1)))
        return y + 56

    def sv_square(self):
        if self.sv_cache[0] != self.hsv[0]:
            self.sv_cache = (self.hsv[0], make_sv_square(self.hsv[0]))
        return self.sv_cache[1]

    def draw_layer_buttons(self, img):
        s = self.session
        self.put(img, "Layers", 12, 20)
        for i, name in enumerate(LAYERS):
            x = 12 + (i % 5) * 108
            y = 28 + (i // 5) * 44
            on = s.is_on(name)
            key = LAYER_KEYS.get(name, str((i + 1) % 10))
            self.button(img, x, y, 102, 38, f"{key} {name}", "layer", name,
                        active=(name == s.selected), dim=not on, right=16)
            # small on/off box: hides the layer without losing its settings
            bx = x + 84
            by = y + 3
            cv2.rectangle(img, (bx, by), (bx + 14, by + 14), (255, 255, 255), -1)
            cv2.rectangle(img, (bx, by), (bx + 14, by + 14), (90, 90, 90), 1)
            if on:
                cv2.line(img, (bx + 3, by + 7), (bx + 6, by + 11), (40, 160, 40), 2)
                cv2.line(img, (bx + 6, by + 11), (bx + 12, by + 3), (40, 160, 40), 2)
            # put it first so it wins over the button under it
            self.hits.insert(0, (bx - 2, by - 2, bx + 16, by + 16, "toggle", name))
        return 28 + ((len(LAYERS) + 4) // 5) * 44 + 20

    def draw_colours(self, img, y, layer, settings):
        self.put(img, f"Colour ({layer})", 12, y)
        y += 8
        presets = list(PRESETS[layer].items())
        current = tuple(settings["color"])
        preset_colours = [tuple(c) for _, c in presets]
        is_custom = current not in preset_colours

        swatches = presets + [("custom...", None)]
        for i, (colour_name, bgr) in enumerate(swatches):
            x = 12 + i * 77
            if bgr is None:
                if is_custom:
                    fill = tuple(int(v) for v in settings["color"])
                else:
                    fill = (255, 255, 255)
                cv2.rectangle(img, (x, y), (x + 70, y + 30), fill, -1)
                img[y + 24:y + 30, x:x + 70] = cv2.resize(HUE_BAR, (70, 6))
                selected = is_custom or self.picker
                action = "picker"
            else:
                cv2.rectangle(img, (x, y), (x + 70, y + 30), tuple(int(v) for v in bgr), -1)
                selected = current == tuple(bgr)
                action = "color"
            if selected:
                cv2.rectangle(img, (x, y), (x + 70, y + 30), ACCENT, 3)
            else:
                cv2.rectangle(img, (x, y), (x + 70, y + 30), (120, 120, 120), 1)
            self.put(img, colour_name[:12], x, y + 44, 0.34)
            self.hits.append((x, y, x + 70, y + 30, action, bgr))
        return y + 56

    def draw_picker(self, img, y, settings):
        sx = 12
        sy = y
        img[sy:sy + SV_H, sx:sx + SV_W] = self.sv_square()
        img[sy + SV_H + 8:sy + SV_H + 8 + HUE_H, sx:sx + SV_W] = HUE_BAR
        h, sat, val = self.hsv
        cx = sx + int(sat / 255 * (SV_W - 1))
        cy = sy + int((255 - val) / 255 * (SV_H - 1))
        cv2.circle(img, (cx, cy), 6, (255, 255, 255), 2)
        hx = sx + int(h / 179 * (SV_W - 1))
        cv2.rectangle(img, (hx - 2, sy + SV_H + 6), (hx + 2, sy + SV_H + 10 + HUE_H), (40, 40, 40), 2)
        px = sx + SV_W + 20
        cv2.rectangle(img, (px, sy), (px + 90, sy + 60), tuple(int(v) for v in settings["color"]), -1)
        cv2.rectangle(img, (px, sy), (px + 90, sy + 60), (120, 120, 120), 1)
        self.put(img, bgr_to_hex(settings["color"]), px, sy + 80, 0.5)
        self.put(img, "click or drag", px, sy + 102, 0.4, DIM)
        self.hits.append((sx, sy, sx + SV_W, sy + SV_H, "sv", (sx, sy)))
        self.hits.append((sx, sy + SV_H + 4, sx + SV_W, sy + SV_H + 12 + HUE_H, "hue", (sx, sy)))
        return y + SV_H + HUE_H + 22

    def status_message(self):
        if self.prompt == "name":
            return f"Look name: {self.typed}_   (Enter = save, Esc = cancel)"
        if self.prompt == "replace":
            return f"'{self.typed}' exists. Replace it? (y / n)"
        if self.prompt == "delete":
            return f"Delete look '{self.session.look_name}'? (y / n)"
        return self.status

    def draw(self):
        s = self.session
        img = np.full((1000, PANEL_W, 3), BG, np.uint8)
        self.hits = []

        y = self.draw_layer_buttons(img)

        layer = s.selected
        settings = s.settings[layer]
        y = self.draw_colours(img, y, layer, settings)
        if self.picker:
            y = self.draw_picker(img, y, settings)

        if layer in STYLES:
            self.put(img, "Style", 12, y + 14)
            for i, style in enumerate(STYLES[layer]):
                self.button(img, 12 + i * 106, y + 22, 100, 28, style, "style", style,
                            active=settings.get("style") == style)
            y += 62

        if layer in s.hidden:
            hidden = "  (hidden)"
        else:
            hidden = ""
        if layer == "skin":
            k_name = "Strength (foundation) k"
        else:
            k_name = "Strength k"
        y = self.slider(img, y, f"{k_name} = {settings['k']:.2f}{hidden}", settings["k"], "strength")
        if layer == "skin":
            # smoothing has its own strength (X42)
            smooth = settings.get("smooth", 0)
            y = self.slider(img, y, f"Smoothing = {smooth:.2f}", smooth, "smooth")
        if layer == "hair":
            lift = settings.get("lift", HAIR_LIFT)
            y = self.slider(img, y, f"Lighten dark hair = {lift:.2f}", lift, "lift")

        self.put(img, f"Look: {s.look_name}", 12, y + 16)
        look_buttons = [("Next look", "next"), ("Reset", "reset"), ("Save look", "save_look"),
                        ("Delete look", "delete_look")]
        for i, (label, action) in enumerate(look_buttons):
            self.button(img, 12 + i * 134, y + 24, 128, 30, label, action)
        y += 66

        if s.use_unet:
            unet_label = "U-Net: on"
        else:
            unet_label = "U-Net: off"
        views = [("Split", "split", self.split), ("Side by side", "side", self.side),
                 ("Masks", "masks", self.masks),
                 (unet_label, "unet", s.use_unet)]
        for i, (label, action, on) in enumerate(views):
            self.button(img, 12 + i * 108, y, 102, 30, label, action, active=on)
        y += 38

        self.put(img, "Detection", 12, y + 20)
        for i, (label, action, on) in enumerate([("Box", "box", self.show_box), ("Dots", "dots", self.show_dots),
                                                  ("Label", "label", self.show_label)]):
            self.button(img, 100 + i * 108, y, 102, 30, label, action, active=on)
        y += 38

        self.button(img, 12, y, 210, 30, "Save image", "save_image")
        self.button(img, PANEL_W - 12 - 130, y, 130, 30, "Quit", "quit")
        y += 40

        if self.prompt:
            msg_colour = (20, 20, 160)
        else:
            msg_colour = (60, 60, 60)
        self.put(img, self.status_message()[:80], 12, y + 14, 0.42, msg_colour)
        return img[:y + 26]


    def click(self, x, y):
        if self.prompt:
            return self.status
        for x0, y0, x1, y1, action, arg in self.hits:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if action in ("strength", "smooth", "lift", "sv", "hue"):
                    self.drag_act = action
                else:
                    self.drag_act = None
                return self.do(action, arg, x, y)
        return self.status

    def drag(self, x, y):
        for hit in self.hits:
            action, arg = hit[4], hit[5]
            if action == self.drag_act:
                self.do(action, arg, x, y)
                return

    def pick_colour(self):
        h, sat, val = self.hsv
        bgr = cv2.cvtColor(np.uint8([[[h, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
        self.session.set_color(tuple(int(v) for v in bgr))

    def open_picker(self):
        """Start the picker at the current colour."""
        colour = self.session.settings[self.session.selected]["color"]
        h, sat, val = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_BGR2HSV)[0, 0]
        self.hsv = (int(h), int(sat), int(val))

    def do(self, act, arg=None, x=None, y=None):
        s = self.session
        if act == "layer":
            s.set_layer(arg)
            self.status = f"{arg} selected"
            if self.picker:
                self.open_picker()
        elif act == "toggle":
            s.toggle_layer(arg)
            if s.is_on(arg):
                self.status = f"{arg} on"
            else:
                self.status = f"{arg} hidden"
        elif act == "color":
            s.set_color(arg)
            self.status = f"{s.selected} colour set"
        elif act == "picker":
            self.picker = not self.picker
            if self.picker:
                self.open_picker()
        elif act == "sv":
            sx, sy = arg
            sat = int(np.clip((x - sx) / (SV_W - 1), 0, 1) * 255)
            val = int(255 - np.clip((y - sy) / (SV_H - 1), 0, 1) * 255)
            self.hsv = (self.hsv[0], sat, val)
            self.pick_colour()
            self.status = f"{s.selected} colour {bgr_to_hex(s.settings[s.selected]['color'])}"
        elif act == "hue":
            sx, _ = arg
            hue = int(np.clip((x - sx) / (SV_W - 1), 0, 1) * 179)
            self.hsv = (hue, self.hsv[1], self.hsv[2])
            self.pick_colour()
            self.status = f"{s.selected} colour {bgr_to_hex(s.settings[s.selected]['color'])}"
        elif act == "style":
            s.set_style(arg)
            self.status = f"{s.selected} style: {arg}"
        elif act == "strength":
            s.set_strength(slider_value(x, arg[0], arg[1]))
        elif act == "smooth":
            s.set_smooth(slider_value(x, arg[0], arg[1]))
        elif act == "lift":
            s.set_lift(slider_value(x, arg[0], arg[1]))
        elif act == "next":
            s.next_look()
            self.status = f"look: {s.look_name}"
        elif act == "reset":
            s.reset()
            self.status = f"look '{s.look_name}' reset"
        elif act == "save_look":
            self.prompt = "name"
            self.typed = ""
        elif act == "delete_look":
            if s.look_name in BUILTIN_LOOKS:
                self.status = "built-in looks can't be deleted"
            else:
                self.prompt = "delete"
        elif act == "split":
            self.split = not self.split
            if self.split:
                self.side = False
        elif act == "side":
            self.side = not self.side
            if self.side:
                self.split = False
        elif act == "masks":
            self.masks = not self.masks
        elif act == "box":
            self.show_box = not self.show_box
        elif act == "dots":
            self.show_dots = not self.show_dots
        elif act == "label":
            self.show_label = not self.show_label
        elif act == "unet":
            s.use_unet = not s.use_unet and s.pipe.par is not None
            if s.use_unet:
                self.status = "masks from the U-Net"
            else:
                self.status = "masks from the landmark polygons"
        elif act == "save_image":
            self.save_request = True
        elif act == "quit":
            self.quit = True
        return self.status

    def key_while_naming(self, k, ch):
        s = self.session
        if k == 27:
            self.prompt = None
            self.status = "save cancelled"
        elif k in (13, 10):
            result = s.save_look(self.typed)
            if result == "exists":
                self.prompt = "replace"
            elif result == "saved":
                self.prompt = None
                self.status = f"saved look '{self.typed}'"
            elif result == "builtin":
                self.status = "that is a built-in look; choose another name"
                self.prompt = "name"
                self.typed = ""
            else:
                self.prompt = None
                self.status = "type a name first"
        elif k in (8, 127):
            self.typed = self.typed[:-1]
        elif ch and (ch.isalnum() or ch in " -_") and len(self.typed) < 30:
            self.typed += ch

    def key(self, k):
        """Handle one key code. Returns False to quit."""
        if self.quit:
            return False
        if k < 0:
            return True
        s = self.session
        if (k & 0xFF) < 128:
            ch = chr(k & 0xFF)
        else:
            ch = ""

        if self.prompt == "name":
            self.key_while_naming(k, ch)
            return True

        if self.prompt == "replace":
            if ch and ch in "yY":
                s.save_look(self.typed, overwrite=True)
                self.status = f"replaced look '{self.typed}'"
            else:
                self.status = "not replaced"
            self.prompt = None
            return True

        if self.prompt == "delete":
            if ch and ch in "yY":
                name = s.look_name
                s.delete_look(name)
                self.status = f"deleted look '{name}'"
            else:
                self.status = "not deleted"
            self.prompt = None
            return True

        if k == 27 or (ch and ch in "qQ"):
            return False
        if ch in KEYS:
            self.do("layer", KEYS[ch])
        elif ch and ch.upper() in LAYER_KEYS.values():
            for layer, letter in LAYER_KEYS.items():
                if letter == ch.upper():
                    self.do("layer", layer)
                    break
        else:
            action = ACTION_KEYS.get(ch.lower())
            if action:
                self.do(action)
        return True


def draw_label(img, text, x):
    # black outline + white text so it can be read on any background
    cv2.putText(img, text, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


CLASS_COLOURS = [  # BGR, one per parser class (facekit.layout.CLASS_NAMES)
    (0, 0, 0), (170, 200, 240), (0, 140, 255), (0, 200, 255), (255, 150, 0), (255, 220, 0),
    (80, 200, 80), (60, 60, 255), (20, 20, 120), (180, 60, 220), (60, 160, 60),
]
SHORT_NAMES = ["background", "skin", "brow L", "brow R", "eye L", "eye R", "nose", "upper lip", "mouth",
               "lower lip", "hair"]


def draw_class_labels(img, labels, text=True):
    """Tint each of the 11 parser classes and write its name at the middle of the area."""
    labels = np.asarray(labels)
    if labels.shape[:2] != img.shape[:2]:
        labels = cv2.resize(labels.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    colour = np.array(CLASS_COLOURS, np.uint8)[labels]
    shown = (labels > 0)[..., None]
    out = np.where(shown, (img * 0.55 + colour * 0.45).astype(np.uint8), img)
    if text:
        for c in range(1, len(SHORT_NAMES)):
            ys, xs = np.nonzero(labels == c)
            if len(xs) < 30:
                continue
            pos = (int(xs.mean()) - 4 * len(SHORT_NAMES[c]), int(ys.mean()))
            cv2.putText(out, SHORT_NAMES[c], pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, SHORT_NAMES[c], pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_detection(img, res, box=True, dots=True, label=True, r=2):
    """Face box, the 106 landmark dots and the 11-class label areas, each only if asked for."""
    out = img
    if label and res.get("labels") is not None:
        out = draw_class_labels(out, res["labels"])
    if box and res.get("box") is not None:
        out = draw_box(out, res["box"], (0, 255, 0), 2)
    if dots and res.get("points") is not None:
        out = draw_points(out, res["points"], r=r)
    return out


def detection_line(res, box=True, dots=True, label=True):
    """Terminal text with only the parts that are switched on ('' if none)."""
    parts = []
    if box:
        if res.get("box") is None:
            parts.append("no face")
        else:
            parts.append("box=(%d,%d,%d,%d) score=%.2f" % (*[int(round(v)) for v in res["box"]], res["score"]))
    if dots:
        parts.append(f"landmarks={0 if res.get('points') is None else len(res['points'])}")
    if label:
        lab = res.get("labels")
        if lab is None:
            parts.append("labels: none")
        else:
            counts = np.bincount(np.asarray(lab).ravel(), minlength=len(SHORT_NAMES))
            total = counts[1:].sum() or 1
            parts.append("labels: " + ", ".join(f"{SHORT_NAMES[c]} {100 * counts[c] / total:.0f}%"
                                                 for c in range(1, len(SHORT_NAMES)) if counts[c]))
    return " | ".join(parts)


def compose(res, panel, fps=None):
    """Make the camera view picture for the current view settings."""
    before = res["before"]
    after = res["after"]
    if panel.masks and res["masks"]:
        view = before.astype(np.float32) * 0.5
        for name, alpha in res["masks"].items():
            a = alpha[..., None]
            view = view * (1 - a) + np.array(MASK_COLOURS[name], np.float32) * a
        view = np.clip(view, 0, 255).astype(np.uint8)
    elif panel.side:
        b = before.copy()
        a = after.copy()
        a = draw_detection(a, res, panel.show_box, panel.show_dots, panel.show_label)
        draw_label(b, "Before", 10)
        draw_label(a, "After", 10)
        view = np.hstack([b, np.full((b.shape[0], 4, 3), 255, np.uint8), a])
    elif panel.split:
        w = before.shape[1] // 2
        view = np.hstack([before[:, :w], after[:, w:]])
        cv2.line(view, (w, 0), (w, view.shape[0]), (255, 255, 255), 2)
        draw_label(view, "Before", 8)
        draw_label(view, "After", w + 8)
    else:
        view = after.copy()

    if not panel.side:
        view = draw_detection(view, res, panel.show_box, panel.show_dots, panel.show_label)
    if not res["face"]:
        cv2.putText(view, "no face", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if fps:
        cv2.putText(view, f"{fps:.0f} FPS", (view.shape[1] - 90, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return view


def pad_to_height(img, h, centre):
    """Fill up to height h with the panel background colour."""
    if centre:
        top = (h - img.shape[0]) // 2
    else:
        top = 0
    out = np.full((h, img.shape[1], 3), BG, np.uint8)
    out[top:top + img.shape[0]] = img
    return out


def combine(view, panel_img):
    """Put the camera view on the left and the controls on the right."""
    h = max(panel_img.shape[0], MIN_H)
    # only the camera part gets scaled so the panel click positions stay the same.
    # the width is capped too (Side by side is very wide)
    scale = min(h / view.shape[0], (MAX_W - GAP - PANEL_W) / view.shape[1])
    size = (round(view.shape[1] * scale), round(view.shape[0] * scale))
    if size != (view.shape[1], view.shape[0]):
        if scale < 1:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        view = cv2.resize(view, size, interpolation=interp)

    panel_x0 = view.shape[1] + GAP
    divider = np.full((h, GAP, 3), 255, np.uint8)
    combined = np.hstack([pad_to_height(view, h, True), divider, pad_to_height(panel_img, h, False)])
    return combined, panel_x0


def save_images(session, res, save_dir, stem=None):
    """13.4: save one Before | After image plus the After image on its own."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = time.strftime("makeup_%Y%m%d_%H%M%S")
    compare_path = save_dir / f"{stem}_compare.png"
    after_path = save_dir / f"{stem}_after.png"
    cv2.imwrite(str(compare_path), session.compare_image(res["before"], res["after"]))
    cv2.imwrite(str(after_path), res["after"])
    return compare_path, after_path


def build_session(args):
    """Load the models named in best_models.json. Missing models only give a warning, so the app always starts."""
    from facekit.pipeline import FacePipeline

    pipe = FacePipeline(args.config, args.device)
    try:
        shown = pipe.config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(pipe.config_path)
    stages = {k: pipe.cfg.get(k, {}).get("stage", "missing").split(" (")[0] for k in ("detector", "landmark", "parser")}
    print(f"models: {shown} ({', '.join(f'{k} {v}' for k, v in stages.items())})")
    session = MakeupSession(pipe, hair_parser=pipe.hair_par, look=args.look, looks_file=args.looks_file)
    session.warnings = pipe.warnings + session.warnings
    return session


def run_photo(args, session):
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    res = session.process(img, track=False)
    line = detection_line(res, args.show_box, args.show_dots, args.show_label)
    if line:
        print(line)
    if not res["face"]:
        print("no face found; the comparison shows the photo unchanged")

    if args.out:
        out = Path(args.out)
    else:
        out = Path(f"outputs/app/{Path(args.image).stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_detection(res["after"], res, args.show_box, args.show_dots, args.show_label))
    compare_path = out.with_name(f"{out.stem}_compare{out.suffix}")
    cv2.imwrite(str(compare_path), session.compare_image(res["before"], res["after"], vertical=args.vertical))
    print(f"after: {out}\ncompare: {compare_path}")
    return 0


def window_closed(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


now_t = time.perf_counter


def run_camera(args, session):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera} (on macOS, allow camera access for the terminal)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    panel = ControlsPanel(session, Path(args.save_dir))
    panel.show_box, panel.show_dots, panel.show_label = args.show_box, args.show_dots, args.show_label
    cv2.namedWindow("Makeup")
    panel_x0 = [640 + GAP]

    def on_mouse(event, x, y, flags, _):
        px = x - panel_x0[0]
        if event == cv2.EVENT_LBUTTONDOWN and px >= 0:
            panel.click(px, y)
        elif event == cv2.EVENT_MOUSEMOVE and panel.drag_act and flags & cv2.EVENT_FLAG_LBUTTON:
            panel.drag(px, y)
        elif event == cv2.EVENT_LBUTTONUP:
            panel.drag_act = None

    cv2.setMouseCallback("Makeup", on_mouse)
    fps = None
    last_time = time.perf_counter()
    shown = False
    last_state, last_print = None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror it like a real mirror
        res = session.process(frame, track=True, want_masks=panel.masks)
        line = detection_line(res, panel.show_box, panel.show_dots, panel.show_label)
        if line and (res["face"] != last_state or now_t() - last_print > 1.0):
            print(line, flush=True)
            last_print = now_t()
        last_state = res["face"]

        now = time.perf_counter()
        frame_fps = 1 / max(now - last_time, 1e-6)
        if fps is None:
            fps = frame_fps
        else:
            fps = 0.9 * fps + 0.1 * frame_fps
        last_time = now

        if panel.save_request:
            compare_path, _ = save_images(session, res, panel.save_dir)
            panel.status = f"saved {compare_path}"
            panel.save_request = False

        img, panel_x0[0] = combine(compose(res, panel, fps), panel.draw())
        cv2.imshow("Makeup", img)
        if not panel.key(cv2.waitKey(1)) or panel.quit:
            break
        if shown and window_closed("Makeup"):
            break
        shown = True
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--look", default="soft pink")
    ap.add_argument("--image", help="photo mode: make up this photo and save it")
    ap.add_argument("--out", help="photo mode: output file (the comparison goes next to it as <name>_compare)")
    ap.add_argument("--vertical", action="store_true", help="photo mode: Before above After (portrait photos)")
    ap.add_argument("--show-box", action="store_true", help="show the face detection box (window and terminal)")
    ap.add_argument("--show-dots", action="store_true", help="show the landmark dots (window and terminal)")
    ap.add_argument("--show-label", action="store_true", help="show the face label and score (window and terminal)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save-dir", default="outputs/app")
    ap.add_argument("--looks-file", default="looks/user_looks.json")
    ap.add_argument("--config", default=None,
                    help="which trained models to use (default: model/best_models.json, the fully trained models, "
                         "or best_models.json, the short-run models, if full training has not been run)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--list-looks", action="store_true")
    args = ap.parse_args(argv)

    session = build_session(args)
    for w in session.warnings:
        print("warning:", w)
    if args.list_looks:
        print("\n".join(session.list_looks()))
        return 0
    if args.image:
        return run_photo(args, session)
    return run_camera(args, session)


if __name__ == "__main__":
    sys.exit(main())
