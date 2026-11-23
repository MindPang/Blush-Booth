import copy
import json

import cv2
import numpy as np
import pytest

from facekit import makeup as MK
from facekit.datasets import parser_matrix
from facekit.geometry import face_box
from tests.fake_lapa import draw_sample


class StubPipeline:
    parser_size = 256

    def __init__(self, img, pts, lab):
        self.img, self.pts, self.lab, self.par = img, pts, lab, object()

    def detect(self, frame):
        return face_box(self.pts), 0.9

    def landmarks_from_box(self, frame, box, passes=2):
        return self.pts.copy()

    def landmarks_from_points(self, frame, prev):
        return self.pts.copy()

    def parse(self, frame, pts, pad=0.2, model=None):
        return self.lab.copy(), parser_matrix(face_box(pts), np.full(4, pad), 0.0, 256)


@pytest.fixture
def face():
    return draw_sample(np.random.RandomState(3), (360, 300))


@pytest.fixture
def session(face, tmp_path):
    return MK.MakeupSession(StubPipeline(*face), looks_file=tmp_path / "looks" / "user_looks.json")


def off_look():
    look = copy.deepcopy(MK.LOOKS["soft pink"])
    for s in look.values():
        s["k"] = 0.0
    look["skin"]["smooth"] = 0.0  # skin smoothing has its own strength (X42)
    return look


def test_builtin_looks_are_valid():
    for name, look in MK.LOOKS.items():
        assert MK.validate_look(look) == [], name
    assert MK.ORDER[0] == "hair" and MK.ORDER[-4:] == ["liner", "lashes", "beard", "mustache"]
    assert sorted(MK.ORDER) == sorted(MK.LAYERS) and len(MK.KEYS) == 10


def test_all_layers_off_changes_nothing(face):
    img, pts, lab = face
    out, _ = MK.paint(img, pts, lab, None, off_look())
    assert np.array_equal(out, img)


def test_each_layer_changes_only_near_its_mask(face):
    img, pts, lab = face
    for name in MK.LAYERS:
        look = off_look()
        look[name]["k"] = 1.0
        if name == "hair":
            look[name]["color"] = (200, 50, 50)
        out, alphas = MK.paint(img, pts, lab, lab, look, want_masks=True)
        changed = np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3
        if name in alphas:
            near = alphas[name] > 0
            assert not (changed & ~near).any(), name
        else:
            assert not changed.any(), name


def test_lips_polygon_mode_and_unet_mode_both_work(face):
    img, pts, lab = face
    look = off_look()
    look["lips"]["k"] = 1.0
    a, _ = MK.paint(img, pts, lab, None, look, use_unet=True)
    b, _ = MK.paint(img, pts, lab, None, look, use_unet=False)
    assert not np.array_equal(a, img) and not np.array_equal(b, img)


def test_session_layer_controls(session):
    session.settings["lens"]["k"] = 0.0
    session.set_layer("lens")
    assert session.selected == "lens" and session.settings["lens"]["k"] == pytest.approx(0.2)  # ON_K (X41)
    session.toggle_layer("lens")
    assert not session.is_on("lens") and session.settings["lens"]["k"] == pytest.approx(0.2)
    session.toggle_layer("lens")
    assert session.is_on("lens")
    session.set_style("wing", "liner")
    with pytest.raises(ValueError):
        session.set_style("nope", "liner")
    session.set_strength(1.7)
    assert session.settings["lens"]["k"] == 1.0
    res = session.process(session.pipe.img, want_masks=True)
    assert res["face"] and res["after"].shape == res["before"].shape and "lens" in res["masks"]


def test_save_load_delete_looks(session, tmp_path):
    f = session.looks_file
    session.set_color((10, 20, 30), "lips")
    assert session.save_look("soft pink") == "builtin"
    assert session.save_look("") == "empty"
    assert session.save_look("mine") == "saved"
    data = json.loads(f.read_text())
    assert data["looks"]["mine"]["lips"]["color"] == "#1E140A"  # hex RGB in the file
    assert session.save_look("mine") == "exists"
    assert session.save_look("mine", overwrite=True) == "saved"
    s2 = MK.MakeupSession(session.pipe, looks_file=f)
    assert s2.list_looks()[:2] == list(MK.BUILTIN_LOOKS) and "mine" in s2.list_looks()
    assert s2.looks["mine"]["lips"]["color"] == (10, 20, 30)
    assert s2.delete_look("soft pink") == "builtin"
    assert s2.delete_look("mine") == "deleted" and "mine" not in json.loads(f.read_text())["looks"]


def test_broken_saved_look_is_skipped_with_warning(face, tmp_path):
    f = tmp_path / "user_looks.json"
    good = MK.look_to_json(MK.LOOKS["natural"])
    bad = copy.deepcopy(good)
    del bad["hair"]
    f.write_text(json.dumps({"looks": {"ok": good, "broken": bad, "natural": good}}))
    s = MK.MakeupSession(StubPipeline(*face), looks_file=f)
    assert "ok" in s.looks and "broken" not in s.looks
    assert any("broken" in w for w in s.warnings) and any("built-in" in w for w in s.warnings)


def test_compare_image_layout(face):
    img = face[0]
    c = MK.compare_image(img, img, "soft pink")
    assert c.shape[1] == 2 * img.shape[1] and c.shape[0] > img.shape[0]
    v = MK.compare_image(img, img, "soft pink", vertical=True)
    assert v.shape[0] > 2 * img.shape[0] and v.shape[1] == img.shape[1]


def test_controls_panel_clicks_and_keys(session, tmp_path):
    import makeup_app as app
    panel = app.ControlsPanel(session, tmp_path)
    img = panel.draw()
    assert img.shape[1] == app.PANEL_W and 300 < img.shape[0] < 1000

    def centre(act, arg=None):
        for x0, y0, x1, y1, a, g in panel.hits:
            if a == act and (arg is None or g == arg):
                return (x0 + x1) // 2, (y0 + y1) // 2
        raise AssertionError((act, arg))

    panel.click(*centre("layer", "blush"))
    assert session.selected == "blush"
    panel.draw()
    sw = list(MK.PRESETS["blush"].values())[2]
    panel.click(*centre("color", sw))
    assert tuple(session.settings["blush"]["color"]) == sw
    panel.draw()
    x0, x1 = next(g for *_, a, g in panel.hits if a == "strength")
    panel.click(x0 + (x1 - x0) // 4, centre("strength")[1])
    assert session.settings["blush"]["k"] == pytest.approx(0.25, abs=0.01)
    panel.draw()
    panel.click(*centre("toggle", "blush"))
    assert not session.is_on("blush")
    panel.do("save_look")
    for ch in "my look":
        panel.key(ord(ch))
    panel.key(13)
    assert "my look" in session.looks and panel.prompt is None
    assert panel.key(ord("q")) is False


def test_new_default_strengths_x41():
    for look in MK.LOOKS.values():
        assert look["hair"]["k"] == 0.29 and look["blush"]["k"] == 0.14 and look["lens"]["k"] == 0.20


def test_panel_style_row_picker_side_by_side_quit(session, tmp_path, face):
    import makeup_app as app
    panel = app.ControlsPanel(session, tmp_path)
    acts = lambda: [h[4] for h in panel.hits]
    session.set_layer("skin")
    panel.draw()
    assert "style" not in acts()
    session.set_layer("lips")
    panel.draw()
    assert "style" in acts()
    panel.do("picker")
    panel.draw()
    sv = next(h for h in panel.hits if h[4] == "sv")
    hue = next(h for h in panel.hits if h[4] == "hue")
    panel.click(hue[0] + 1, (hue[1] + hue[3]) // 2)
    panel.click(sv[2] - 1, sv[1] + 1)
    b, g, r = session.settings["lips"]["color"]
    assert r > 240 and g < 15 and b < 15
    panel.draw()
    assert "picker" in acts() and panel.picker
    img = face[0]
    res = {"before": img, "after": img, "masks": {}, "points": None, "face": True}
    panel.do("split")
    panel.do("side")
    assert panel.side and not panel.split
    assert app.compose(res, panel).shape[1] == 2 * img.shape[1] + 4
    panel.draw()
    x0, y0, x1, y1, *_ = next(h for h in panel.hits if h[4] == "quit")
    panel.click((x0 + x1) // 2, (y0 + y1) // 2)
    assert panel.quit and panel.key(-1) is False


def _skin_only(face, k, smooth):
    img, pts, lab = face
    look = off_look()
    look["skin"]["k"], look["skin"]["smooth"] = k, smooth
    return MK.paint(img, pts, lab, None, look, want_masks=True)


def test_skin_smoothing_has_its_own_strength(face):
    img, pts, lab = face
    rng = np.random.RandomState(0)
    noisy = np.clip(img.astype(int) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    f = (noisy, pts, lab)
    out0, _ = _skin_only(f, 0.0, 0.0)
    assert np.array_equal(out0, noisy)
    out_s, al = _skin_only(f, 0.0, 1.0)
    changed = np.abs(out_s.astype(int) - noisy.astype(int)).sum(-1) > 3
    assert changed.any() and not (changed & ~(al["skin"] > 0)).any()
    skin = lab == 1
    assert out_s[skin].std(0).mean() < noisy[skin].std(0).mean()
    ab = lambda im: MK.to_lab(im.astype(np.float32))[skin][:, 1:].mean(0)
    assert np.abs(ab(out_s) - ab(noisy)).max() < 3
    out_k, _ = _skin_only(f, 1.0, 0.0)
    assert not np.array_equal(out_k, out_s)


def test_set_smooth_on_off_and_old_looks(session, tmp_path, face):
    session.set_strength(0.0, "skin")
    session.set_smooth(0.5)
    assert session.is_on("skin")
    session.toggle_layer("skin")
    assert not session.is_on("skin") and session.settings["skin"]["smooth"] == 0.5
    f = tmp_path / "old.json"
    old = MK.look_to_json(MK.LOOKS["natural"])
    del old["skin"]["smooth"]
    f.write_text(json.dumps({"looks": {"old": old}}))
    s2 = MK.MakeupSession(StubPipeline(*face), looks_file=f)
    assert s2.looks["old"]["skin"]["smooth"] == s2.looks["old"]["skin"]["k"]


def test_panel_smoothing_slider_only_for_skin(session, tmp_path):
    import makeup_app as app
    panel = app.ControlsPanel(session, tmp_path)
    session.set_layer("lips")
    panel.draw()
    assert "smooth" not in [h[4] for h in panel.hits]
    session.set_layer("skin")
    panel.draw()
    x0, y0, x1, y1, act, (a, b) = next(h for h in panel.hits if h[4] == "smooth")
    panel.click(a + (b - a) * 3 // 4, (y0 + y1) // 2)
    assert session.settings["skin"]["smooth"] == pytest.approx(0.75, abs=0.01)


def _with(look_mods, face):
    img, pts, lab = face
    look = off_look()
    for name, mods in look_mods.items():
        look[name].update(mods)
    return MK.paint(img, pts, lab, lab, look, want_masks=True)


def test_mustache_is_off_by_default_and_wig_is_gone(session):
    assert "wig" not in MK.LAYERS and "wig" not in MK.ORDER and "wig" not in MK.STYLES
    for look in MK.LOOKS.values():
        assert look["mustache"]["k"] == 0 and "wig" not in look
    assert not session.is_on("mustache")
    session.set_layer("mustache")
    assert session.is_on("mustache") and session.settings["mustache"]["k"] == pytest.approx(0.90)
    session.set_style("pencil")
    with pytest.raises(ValueError):
        session.set_style("bob")
    session.toggle_layer("mustache")
    assert not session.is_on("mustache")


def test_mustache_is_lifted_off_the_lip(face, monkeypatch):
    img, pts, lab = face
    centre_y = lambda out: np.nonzero(np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3)[0].mean()
    lifted = centre_y(_with({"mustache": {"k": 1.0}}, face)[0])
    monkeypatch.setattr(MK, "MUSTACHE_LIFT", 0.0)
    before = centre_y(_with({"mustache": {"k": 1.0}}, face)[0])
    half_mouth = (pts[90, 0] - pts[84, 0]) / 2
    assert before - lifted > 0.5 * 0.10 * half_mouth


def test_mustache_not_squashed_when_nose_is_level_with_mouth(face):
    img, pts, lab = face
    flat = pts.copy()
    flat[60] = [flat[60, 0], (flat[84, 1] + flat[90, 1]) / 2]
    out, _ = MK.paint(img, flat, lab, lab, {**off_look(), "mustache": {**MK.LOOKS["natural"]["mustache"], "k": 1.0}})
    ys = np.nonzero(np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3)[0]
    assert ys.max() - ys.min() > 0.15 * (pts[90, 0] - pts[84, 0])


def test_mustache_sits_between_nose_and_mouth(face):
    img, pts, lab = face
    for style in MK.STYLES["mustache"]:
        out, al = _with({"mustache": {"k": 1.0, "style": style}}, face)
        ys, xs = np.nonzero(np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3)
        assert len(ys) > 30, style
        mouth_w = pts[90, 0] - pts[84, 0]
        mid = (xs > pts[84, 0]) & (xs < pts[90, 0])
        assert ys[mid].min() >= pts[55:66, 1].min() - 2 and ys.max() <= pts[84:104, 1].max(), style
        assert ys.min() >= pts[55:66, 1].min() - 0.25 * mouth_w, style
        assert xs.min() >= pts[84, 0] - mouth_w and xs.max() <= pts[90, 0] + mouth_w, style


def test_old_saved_looks_with_or_without_wig_load(face, tmp_path):
    f = tmp_path / "old.json"
    before = MK.look_to_json(MK.LOOKS["natural"])
    del before["mustache"]                                   # saved before X46
    with_wig = MK.look_to_json(MK.LOOKS["natural"])
    with_wig["wig"] = {"color": "#463C1E", "k": 0.95, "blur": 3, "style": "bob"}  # saved during X46-X47
    f.write_text(json.dumps({"looks": {"old": before, "wiggy": with_wig}}))
    s2 = MK.MakeupSession(StubPipeline(*face), looks_file=f)
    assert s2.looks["old"]["mustache"]["k"] == 0 and "wig" not in s2.looks["wiggy"] and not s2.warnings


def test_panel_mustache_button_and_key(session, tmp_path):
    import makeup_app as app
    panel = app.ControlsPanel(session, tmp_path)
    panel.draw()
    assert {g for *_, a, g in panel.hits if a == "layer"} == set(MK.LAYERS)
    panel.key(ord("t"))
    assert session.selected == "mustache" and session.is_on("mustache")
    panel.draw()
    assert [g for *_, a, g in panel.hits if a == "style"] == MK.STYLES["mustache"]


def test_teeth_follow_the_chosen_colour():  # X49: teeth used to ignore the colour (always white)
    rng = np.random.RandomState(0)
    sub = np.clip(rng.normal((170, 200, 215), 8, (20, 30, 3)), 0, 255).astype(np.float32)  # yellowish teeth (BGR)
    m = np.ones(sub.shape[:2], bool)
    look = MK.LOOKS["soft pink"]["teeth"]
    mean = lambda col: MK.layer_colour("teeth", {**look, "color": col}, sub, m, None, 200).reshape(-1, 3).mean(0)
    b, g, r = mean((0, 0, 255))
    assert r > g + 40 and r > b + 40
    b, g, r = mean((255, 0, 0))
    assert b > r + 40
    w = mean((245, 245, 245))
    assert w.min() > sub.reshape(-1, 3).mean(0).min() and np.ptp(w) < np.ptp(sub.reshape(-1, 3).mean(0))


def test_mustache_hidden_when_lips_not_seen(face, monkeypatch):  # X50
    img, pts, lab = face
    monkeypatch.setattr(MK, "LIP_SEEN", (0.3, 0.6))
    look = {**off_look(), "mustache": {**MK.LOOKS["natural"]["mustache"], "k": 1.0}}
    shown = lambda labels, unet=True: not np.array_equal(MK.paint(img, pts, labels, labels, look, use_unet=unet)[0], img)
    assert MK.lip_fraction(pts, lab) > 0.6 and shown(lab)
    covered = lab.copy()
    covered[MK.poly_mask(lab.shape, pts[84:96])] = 1
    assert MK.lip_fraction(pts, covered) == 0 and not shown(covered)
    assert shown(covered, unet=False)
    half = lab.copy()
    m = MK.poly_mask(lab.shape, pts[84:96])
    m[:, int(pts[84:96, 0].mean()):] = False
    half[m] = 1
    v = MK.lip_visibility(pts, half)
    assert 0 < v < 1 or v == 1


def test_eye_layers_hidden_on_covered_eye(face, monkeypatch):  # X51 liner, lashes; X53 shadow, lens
    img, pts, lab = face
    monkeypatch.setattr(MK, "EYE_SEEN", (0.3, 0.6))
    mid_x = (pts[MK.EYE["R"]["pupil"], 0] + pts[MK.EYE["L"]["pupil"], 0]) / 2
    assert set(MK.EYE_LAYERS) == {"shadow", "lens", "liner", "lashes"}
    for name in MK.EYE_LAYERS:
        look = off_look()
        look[name]["k"] = 1.0

        def changed(labels, unet=True):
            out, _ = MK.paint(img, pts, labels, labels, look, use_unet=unet)
            xs = np.nonzero(np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3)[1]
            return (xs < mid_x).any(), (xs > mid_x).any()       # (image-left eye side, image-right eye side)

        assert min(MK.eye_fraction(pts, lab, s) for s in "RL") > 0.6
        assert changed(lab) == (True, True), name
        one = lab.copy()
        one[MK.poly_mask(lab.shape, MK.eye_polygon(pts, "R"))] = 1  # parser sees skin over the image-left eye
        assert MK.eye_visibility(pts, one) == {"R": 0.0, "L": 1.0}
        assert changed(one) == (False, True), name
        both = one.copy()
        both[MK.poly_mask(lab.shape, MK.eye_polygon(pts, "L"))] = 1
        assert changed(both) == (False, False), name
        assert changed(both, unet=False) == (True, True), name


def test_no_lens_on_closed_eye(face):  # X54
    img, pts, lab = face
    look = off_look()
    look["lens"]["k"] = 1.0
    mid_x = (pts[MK.EYE["R"]["pupil"], 0] + pts[MK.EYE["L"]["pupil"], 0]) / 2
    sides = lambda q: (lambda xs: ((xs < mid_x).any(), (xs > mid_x).any()))(
        np.nonzero(np.abs(MK.paint(img, q, lab, lab, look)[0].astype(int) - img.astype(int)).sum(-1) > 3)[1])
    assert min(MK.iris_visible(pts, s) for s in "RL") > MK.LENS_OPEN[1] and sides(pts) == (True, True)
    shut = pts.copy()
    e = MK.EYE["R"]
    lower = {66: 66, 67: 73, 68: 72, 69: 71, 70: 70}
    for u, l in lower.items():
        shut[u, 1] = shut[l, 1] - 0.5
    assert MK.iris_visible(shut, "R") < MK.LENS_OPEN[0] and MK.lens_open(shut)["R"] == 0
    assert sides(shut) == (False, True)


def _hair_only(face, color, k=0.29, lift=None):
    img, pts, lab = face
    look = off_look()
    look["hair"].update(color=color, k=k)
    if lift is None:
        look["hair"].pop("lift", None)
    else:
        look["hair"]["lift"] = lift
    return MK.paint(img, pts, lab, None, look)[0]


def _hair_L(img, lab):
    return float(np.median(MK.to_lab(img.astype(np.float32))[..., 0][lab == 10]))


def test_blonde_on_black_hair_is_light(face):
    img, pts, lab = face
    blonde = MK.PRESETS["hair"]["blonde"]
    L0 = _hair_L(img, lab)
    no_lift = _hair_L(_hair_only(face, blonde, lift=0.0), lab)
    lifted = _hair_L(_hair_only(face, blonde, lift=MK.HAIR_LIFT), lab)
    assert no_lift < L0 + 40                                 # without the lift, blonde only tints black hair
    assert lifted > L0 + 80 and lifted > no_lift + 60
    assert np.array_equal(_hair_only(face, blonde, lift=None), _hair_only(face, blonde, lift=MK.HAIR_LIFT))
    hair = lab == 10
    out = _hair_only(face, blonde, lift=1.0)
    far = cv2.dilate(hair.astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    assert np.array_equal(out[far], img[far])


def test_lift_never_darkens_and_is_skipped_for_dark_colours(face):
    img, pts, lab = face
    black = MK.PRESETS["hair"]["black"]
    assert np.array_equal(_hair_only(face, black, lift=1.0), _hair_only(face, black, lift=0.0))
    sub = img.astype(np.float32)
    up = MK.lift_hair(sub, lab == 10, MK.PRESETS["hair"]["blonde"], 1.0)
    L0, L1 = MK.to_lab(sub)[..., 0], MK.to_lab(up)[..., 0]
    assert (L1 >= L0 - 1).all()
    brown = _hair_L(_hair_only(face, MK.PRESETS["hair"]["brown"], lift=1.0), lab)
    blonde = _hair_L(_hair_only(face, MK.PRESETS["hair"]["blonde"], lift=1.0), lab)
    assert brown < blonde - 60


def test_lift_setting_validation_session_and_panel(session, tmp_path):
    import makeup_app as app
    look = copy.deepcopy(MK.LOOKS["natural"])
    look["hair"]["lift"] = 1.5
    assert "hair: lift must be 0-1" in MK.validate_look(look)
    session.set_lift(2.0)
    assert session.settings["hair"]["lift"] == 1.0
    panel = app.ControlsPanel(session, tmp_path)
    session.set_layer("lips")
    panel.draw()
    assert "lift" not in [h[4] for h in panel.hits]
    session.set_layer("hair")
    panel.draw()
    x0, y0, x1, y1, act, (a, b) = next(h for h in panel.hits if h[4] == "lift")
    panel.click(a + (b - a) // 4, (y0 + y1) // 2)
    assert session.settings["hair"]["lift"] == pytest.approx(0.25, abs=0.01)


def _changed(out, img):
    return np.abs(out.astype(int) - img.astype(int)).sum(-1) > 3


def test_beard_is_off_by_default_and_separate_from_mustache(session):
    for look in MK.LOOKS.values():
        assert look["beard"]["k"] == 0 and look["beard"]["style"] == "full"
    assert "beard" in MK.OVERLAYS and MK.STYLES["beard"] != MK.STYLES["mustache"]
    assert not session.is_on("beard")
    session.set_layer("beard")
    assert session.is_on("beard") and not session.is_on("mustache")
    assert session.settings["beard"]["k"] == pytest.approx(0.85)
    session.set_style("goatee")
    with pytest.raises(ValueError):
        session.set_style("handlebar")
    session.set_color(MK.PRESETS["beard"]["ginger"])
    assert session.settings["mustache"]["color"] != session.settings["beard"]["color"]
    session.toggle_layer("beard")
    assert not session.is_on("beard")


def test_beard_styles_sit_on_the_lower_face_and_leave_the_lips(face):
    img, pts, lab = face
    fw = MK.face_width(pts)
    lips = MK.poly_mask(lab.shape, pts[84:96])
    areas = {}
    for style in MK.STYLES["beard"]:
        out, al = _with({"beard": {"k": 1.0, "style": style}}, face)
        ch = _changed(out, img)
        ys, xs = np.nonzero(ch)
        assert len(ys) > 100, style
        assert not (ch & lips).any(), style
        assert ys.max() <= pts[16, 1] + 0.15 * fw, style
        assert ys.min() >= pts[0:33, 1].min() - 2, style
        assert xs.min() >= pts[0:33, 0].min() - 0.06 * fw and xs.max() <= pts[0:33, 0].max() + 0.06 * fw, style
        chin = int(round(pts[16, 1])) - 3, int(round(pts[16, 0]))
        assert ch[chin], style
        areas[style] = int(ch.sum())
    assert areas["full"] > areas["goatee"] and areas["full"] > areas["chin strap"]
    near = 84 if abs(pts[84, 0] - pts[8, 0]) < abs(pts[90, 0] - pts[8, 0]) else 90
    cheek = np.round(0.6 * pts[8] + 0.4 * pts[near]).astype(int)
    out_g, _ = _with({"beard": {"k": 1.0, "style": "goatee"}}, face)
    out_f, _ = _with({"beard": {"k": 1.0, "style": "full"}}, face)
    assert _changed(out_f, img)[cheek[1], cheek[0]] and not _changed(out_g, img)[cheek[1], cheek[0]]


def test_beard_and_mustache_together(face):
    img, pts, lab = face
    both, al = _with({"beard": {"k": 1.0}, "mustache": {"k": 1.0}}, face)
    assert set(al) == {"beard", "mustache"}
    only_b, _ = _with({"beard": {"k": 1.0}}, face)
    only_m, _ = _with({"mustache": {"k": 1.0}}, face)
    assert (_changed(both, img) >= (_changed(only_b, img) | _changed(only_m, img))).mean() > 0.99


def test_beard_hidden_where_the_chin_is_covered(face):
    img, pts, lab = face
    look = {**off_look(), "beard": {**MK.LOOKS["natural"]["beard"], "k": 1.0}}
    paint = lambda labels, unet=True: MK.paint(img, pts, labels, labels, look, use_unet=unet)[0]
    assert _changed(paint(lab), img).sum() > 100
    hand = lab.copy()
    hand[MK.poly_mask(lab.shape, pts[0:33])] = 0
    assert _changed(paint(hand), img).sum() == 0
    assert _changed(paint(hand, unet=False), img).sum() > 100
    part = lab.copy()
    fw = MK.face_width(pts)
    c = np.round(0.6 * pts[16] + 0.4 * pts[93]).astype(int)
    r = int(0.08 * fw)
    part[c[1] - r:c[1] + r, c[0] - r:c[0] + r] = 0
    out = paint(part)
    assert _changed(out, img).sum() > 100 and not _changed(out, img)[c[1], c[0]]


def test_old_looks_without_beard_and_panel_key(face, tmp_path, session):
    import makeup_app as app
    f = tmp_path / "old.json"
    old = MK.look_to_json(MK.LOOKS["natural"])
    del old["beard"]
    f.write_text(json.dumps({"looks": {"old": old}}))
    s2 = MK.MakeupSession(StubPipeline(*face), looks_file=f)
    assert s2.looks["old"]["beard"]["k"] == 0 and not s2.warnings
    panel = app.ControlsPanel(session, tmp_path)
    panel.key(ord("j"))
    assert session.selected == "beard" and session.is_on("beard")
    panel.draw()
    assert [g for *_, a, g in panel.hits if a == "style"] == MK.STYLES["beard"]


def test_beard_top_stays_low_on_the_face(face):
    img, pts, lab = face
    ex, down, fw, c = MK.face_axes(pts)
    nose = float((pts[60] - c) @ down)                       # negative: the nose bottom is above the mouth centre
    for style in ("full", "chin strap", "stubble"):
        ch = _changed(_with({"beard": {"k": 1.0, "style": style}}, face)[0], img)
        ys, xs = np.nonzero(ch)
        h = (np.c_[xs, ys] - c) @ down
        assert h.min() >= MK.BEARD_TOP * nose - 0.03 * fw, style
        assert h.min() > 0.6 * nose, style
