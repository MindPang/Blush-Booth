import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import hashlib
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from makeup_app import build_session, save_images  # noqa: E402
from ui.actions import App  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
APP = None
lock = threading.Lock()
print_state = {"face": None, "time": 0.0}
CAMERA_INDEX = None   # set by --camera; when set, the server reads the camera (e.g. an iPhone) instead of the browser
cam = {"cap": None}
cam_lock = threading.Lock()


def short_path(p):
    """Path relative to the project if it's inside it, otherwise the full path."""
    p = Path(p).resolve()
    if p.is_relative_to(ROOT):
        return str(p.relative_to(ROOT))
    return str(p)


def reply(result):
    body, code = result
    return jsonify(body), code


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_file(name):
    return send_from_directory(STATIC, name)


@app.get("/api/state")
def get_state():
    with lock:
        return reply(APP.ok(APP.notice))


@app.post("/api/frame")
def frame():
    """The body is one JPEG/PNG image. Add ?track=1 for camera frames (follows the face between frames)."""
    data = request.get_data()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "could not read image", 400
    track = request.args.get("track") == "1"
    with lock:
        key = None if track else hashlib.sha1(data).digest()
        out, res, line = APP.finish(img, track, key, APP.cached_analysis(key))
        now = time.perf_counter()
        if line and (res["face"] != print_state["face"] or now - print_state["time"] > 1.0):
            print(line, flush=True)
            print_state["time"] = now
        print_state["face"] = res["face"]
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resp = app.response_class(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["X-Face"] = "1" if res["face"] else "0"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/config")
def config():
    return jsonify({"server_camera": CAMERA_INDEX is not None})


@app.get("/api/camera/raw")
def camera_raw():
    """One raw JPEG frame from the server-side camera (opened on first use)."""
    if CAMERA_INDEX is None:
        return "server camera is off, start with --camera N", 404
    with cam_lock:
        if cam["cap"] is None:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                return f"could not open camera {CAMERA_INDEX} (on macOS, allow camera access for the terminal)", 500
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam["cap"] = cap
        ok, img = cam["cap"].read()
    if not ok:
        return "could not read from the camera", 500
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resp = app.response_class(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/camera/release")
def camera_release():
    with cam_lock:
        if cam["cap"] is not None:
            cam["cap"].release()
            cam["cap"] = None
    return "", 204


@app.post("/api/do")
def do():
    """One change, like {"action": "color", "layer": "lips", "hex": "#AA3355"}. Returns the new state."""
    data = request.get_json(force=True) or {}
    with lock:
        return reply(APP.do(data))


def setup(argv=None):
    """Parse the options and load the models. Returns the options."""
    global APP, CAMERA_INDEX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--camera", type=int, default=None,
                    help="read this camera index on the server (like makeup_app.py) instead of the browser camera; use it for an iPhone")
    ap.add_argument("--no-browser", action="store_true", help="don't open the page in the browser")
    ap.add_argument("--look", default="soft pink")
    ap.add_argument("--save-dir", default=str(ROOT / "outputs" / "app"))
    ap.add_argument("--looks-file", default=str(ROOT / "looks" / "user_looks.json"))
    ap.add_argument("--config", default=None,
                    help="which trained models to use (default: model/best_models.json if full training was run, "
                         "else best_models.json)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--show-box", action="store_true", help="show the face detection box (page and terminal)")
    ap.add_argument("--show-dots", action="store_true", help="show the landmark dots (page and terminal)")
    ap.add_argument("--show-label", action="store_true", help="show the 11 class label areas (page and terminal)")
    args = ap.parse_args(argv)
    CAMERA_INDEX = args.camera

    save_dir = Path(args.save_dir)
    print("loading models ...")
    start = time.perf_counter()
    session = build_session(args)
    print(f"ready in {time.perf_counter() - start:.1f} s")
    for w in session.warnings:
        print("warning:", w)

    def save(s, res):
        return [short_path(p) for p in save_images(s, res, save_dir)]

    APP = App(session, save_images=save,
              show={"box": args.show_box, "dots": args.show_dots, "label": args.show_label})
    # shown once on the page when it opens, e.g. "No trained detector yet ..."
    APP.notice = " ".join(w for w in session.warnings if w.startswith("No trained"))
    return args


def main(argv=None):
    args = setup(argv)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  open {url}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
