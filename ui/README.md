# Blush Booth: the browser UI

The same makeup as `makeup_app.py` (it uses `MakeupSession` from `facekit/makeup.py`), with a browser page as the front end instead of OpenCV windows.

```bash
python ui/server.py            # loads the models, then opens http://127.0.0.1:5050
```

Options: `--port 8000`, `--look natural`, `--no-browser`, `--device cpu`, `--config <file>` (which trained models to use; default `model/best_models.json`, the fully trained models, or `best_models.json` if full training was not run) and `--looks-file`.

If no model is trained yet, the page still opens and shows a message that says which step to run.

## How to play
1. Click **Use camera** or **Choose photo**. You can also drop a photo onto the picture.
2. Pick a **Look** from the menu at the top right (Soft pink, Natural, or one you saved).
3. The layers sit in four tabs: **Face** (foundation, blush, brows), **Eyes** (eyeshadow, eyeliner, lashes, lenses), **Lips** (lips, teeth) and **Hair** (hair colour, mustache). The number on a tab shows how many of its layers are on.
4. Each layer has an on/off switch, colour swatches with shade names ("other" picks any colour), a style where it has one, and an Amount slider. Lips have Gloss, eyeshadow has Shimmer, and foundation has Smoothing.
5. **Compare** shows a before/after split you can drag. **Snap** saves `<time>_compare.png` and `<time>_after.png` to `outputs/app/` and offers a download.
6. **Save** stores the current settings as your own look in `looks/user_looks.json`, the same file the OpenCV app uses. **Delete** (click twice) removes a saved look, **Reset** undoes your changes to the look, and **Clear** switches every layer off.

Shortcuts: `Space` snap · `C` compare · `1`–`4` tabs.

## How it works
- `server.py`: Flask. `POST /api/frame` takes a JPEG and returns the made-up JPEG. `POST /api/do` changes a setting and returns the new state. `GET /api/state` returns the current state.
- `static/`: `index.html`, `style.css` and `app.js`, with no build step. The browser mirrors the camera and sends 640-px frames one at a time, so it never piles up. Photos are scaled down to 1280 px.
- The server only listens on 127.0.0.1, so pictures never leave the computer. The page loads its fonts from Google Fonts; without internet it falls back to the system font.
- Speed depends on which models won the experiments (see `best_models.json`).
