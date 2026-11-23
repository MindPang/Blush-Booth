import numpy as np

from facekit.makeup import BUILTIN_LOOKS, HAIR_LIFT, LAYERS, PRESETS, STYLES, bgr_to_hex, hex_to_bgr
from makeup_app import detection_line, draw_detection

EXTRAS = {
    "lips": ("gloss", "Gloss"),
    "shadow": ("shimmer", "Shimmer"),
    "skin": ("smooth", "Smoothing"),
    "hair": ("lift", "Lighten dark hair"),
}


class App:
    """One makeup session plus the bits of page state around it."""

    def __init__(self, session, save_images=None, show=None):
        self.session = session
        self.save_images = save_images
        self.show = {"box": False, "dots": False, "label": False}
        if show:
            self.show.update(show)
        self.last = None
        self.photo = {"key": None, "analysis": None}
        self.notice = ""   # message for the page when it opens (missing models)


    def state(self):
        """Everything the page needs to draw the controls."""
        s = self.session
        layers = []
        for name in LAYERS:
            settings = s.settings[name]
            extra = None
            if name in EXTRAS:
                key, label = EXTRAS[name]
                default = HAIR_LIFT if key == "lift" else 0
                extra = {"key": key, "label": label, "value": float(settings.get(key, default))}
            presets = [{"name": n, "hex": bgr_to_hex(c)} for n, c in PRESETS[name].items()]
            layers.append({
                "name": name,
                "on": s.is_on(name),
                "color": bgr_to_hex(settings["color"]),
                "k": float(settings["k"]),
                "style": settings.get("style"),
                "styles": STYLES.get(name, []),
                "presets": presets,
                "extra": extra,
            })
        return {
            "layers": layers,
            "selected": s.selected,
            "look": s.look_name,
            "looks": s.list_looks(),
            "builtin": list(BUILTIN_LOOKS),
            "unet": bool(s.use_unet),
            "has_unet": s.pipe.par is not None,
            "show": dict(self.show),
        }

    def ok(self, msg="", **extra):
        return {"ok": True, "msg": msg, "state": self.state(), **extra}, 200

    def fail(self, msg, code=400):
        return {"ok": False, "msg": msg, "state": self.state()}, code


    def cached_analysis(self, key):
        """The saved face analysis if `key` (a hash of the photo bytes) is the last photo, else None."""
        if key is not None and self.photo["key"] == key:
            return self.photo["analysis"]
        return None

    def finish(self, img, track, key=None, analysis=None):
        """Put the makeup on."""
        res = self.session.process(img, track=track, analysis=analysis)
        if key is not None:
            self.photo.update(key=key, analysis=res["analysis"])
        else:
            self.photo.update(key=None, analysis=None)
        self.last = res
        show = self.show
        out = draw_detection(res["after"], res, show["box"], show["dots"], show["label"])
        return out, res, detection_line(res, show["box"], show["dots"], show["label"])


    def set_extra(self, layer, key, value):
        """Change the extra slider of a layer (gloss, shimmer, smooth or lift)."""
        s = self.session
        if key == "smooth":
            s.set_smooth(value)
        elif key == "lift":
            s.set_lift(value)
        else:
            s.settings[layer][key] = value
            s.hidden.discard(layer)

    def do(self, data):
        """One change, like {"action": "color", "layer": "lips", "hex": "#AA3355"}.

        Returns (reply dict, HTTP status).
        """
        action = data.get("action")
        layer = data.get("layer")
        s = self.session
        if layer is not None and layer not in LAYERS:
            return self.fail(f"unknown layer {layer!r}")

        if action == "select":
            s.set_layer(layer)

        elif action == "toggle":
            s.toggle_layer(layer)
            return self.ok(f"{layer} {'on' if s.is_on(layer) else 'off'}")

        elif action == "color":
            s.set_color(hex_to_bgr(data["hex"]), layer)

        elif action == "style":
            s.set_style(data["style"], layer)

        elif action == "strength":
            s.set_strength(float(data["value"]), layer)

        elif action == "extra":
            key = data.get("key")
            value = float(np.clip(float(data["value"]), 0, 1))
            if EXTRAS.get(layer, (None,))[0] != key:
                return self.fail(f"{layer} has no {key!r}")
            self.set_extra(layer, key, value)

        elif action == "look":
            if data.get("name") not in s.looks:
                return self.fail("no such look")
            s.load_look(data["name"])
            return self.ok(f"Look: {s.look_name}")

        elif action == "reset":
            s.reset()
            return self.ok("Back to the start of this look")

        elif action == "bare":
            for name in LAYERS:
                if s.is_on(name):
                    s.hidden.add(name)
            return self.ok("All makeup off")

        elif action == "save_look":
            name = str(data.get("name", "")).strip()[:30]
            result = s.save_look(name, overwrite=bool(data.get("overwrite")))
            if result == "exists":
                return {"ok": False, "exists": True, "msg": f"'{name}' already exists", "state": self.state()}, 200
            if result == "saved":
                return self.ok(f"Saved look '{name}'")
            if result == "builtin":
                return self.fail("That name belongs to a built-in look")
            return self.fail("Give your look a name first")

        elif action == "delete_look":
            name = data.get("name", s.look_name)
            result = s.delete_look(name)
            if result == "builtin":
                return self.fail("Built-in looks can't be deleted")
            if result != "deleted":
                return self.fail("No such look")
            return self.ok(f"Deleted look '{name}'")

        elif action == "unet":
            s.use_unet = (not s.use_unet) and s.pipe.par is not None
            return self.ok("Masks from the U-Net" if s.use_unet else "Masks from the landmark outline")

        elif action == "detect":
            part = data.get("part")
            if part not in self.show:
                return self.fail(f"unknown detection part {part!r}")
            self.show[part] = bool(data.get("on", not self.show[part]))
            return self.ok(f"{part} {'on' if self.show[part] else 'off'}")

        elif action == "save_image":
            if self.last is None:
                return self.fail("Nothing to save yet")
            if self.save_images is None:
                return self.ok("Picture ready")
            files = [str(p) for p in self.save_images(s, self.last)]
            return self.ok(f"Saved to {files[0]}", files=files)

        else:
            return self.fail(f"unknown action {action!r}")
        return self.ok()
