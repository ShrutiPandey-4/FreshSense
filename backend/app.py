"""
FreshSense Backend API
Run with: python app.py
Then open  http://localhost:5050  (the backend also serves the website, one command only)

Endpoints
  GET  /api/health
  POST /api/analyze   form-data: image=<file>  [normalize=1]
  POST /api/compare   form-data: image_a=<file>, image_b=<file>  [normalize=1]
  POST /api/compare-models  form-data: image=<file>  [normalize=1]
"""

import os
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from freshness_model import analyze_image, compare_images
from model_comparison import compare_classification_models

VERSION = "freshsense15"
PORT = 5050
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = Flask(__name__)
CORS(app)  # allow the frontend (different port) to call this API

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _wants_normalize():
    return request.form.get("normalize", "0").lower() in ("1", "true", "on", "yes")


def _get_upload(field):
    """Returns (bytes, None) or (None, (error_response, status))."""
    if field not in request.files:
        return None, (jsonify({"error": f"No file '{field}' provided. Send as multipart/form-data."}), 400)
    file = request.files[field]
    if file.filename == "":
        return None, (jsonify({"error": f"Empty filename for '{field}'."}), 400)
    if not allowed_file(file.filename):
        return None, (jsonify({"error": "Unsupported file type. Use png, jpg, jpeg or webp."}), 400)
    data = file.read()
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        return None, (jsonify({"error": f"'{field}' is not a valid or supported image. Please choose a PNG, JPG, JPEG or WebP photo."}), 400)
    return data, None


@app.route("/")
def home():
    resp = send_from_directory(FRONTEND_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store"          # never show an old saved copy of the page
    return resp


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "FreshSense API", "version": VERSION})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data, err = _get_upload("image")
    if err:
        return err
    try:
        if os.environ.get("FRESHSENSE_DEBUG_DETECTION", "1").lower() not in {"0", "false", "off"}:
            print(f"[freshsense][api] accepted image upload: {len(data)} bytes", flush=True)
        result = analyze_image(data, normalize=_wants_normalize())
        if os.environ.get("FRESHSENSE_DEBUG_DETECTION", "1").lower() not in {"0", "false", "off"}:
            fruit_rows = [
                (f.get("id"), f.get("food"), f.get("freshness_score"), f.get("status"), f.get("bbox"))
                for f in result.get("fruits", [])
            ]
            print(
                f"[freshsense][api] response fruit_count={result.get('fruit_count')}, fruits={fruit_rows}",
                flush=True,
            )
        return jsonify(result), 200
    except Exception:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": "Could not analyze this image. Please try a different photo."}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    data_a, err = _get_upload("image_a")
    if err:
        return err
    data_b, err = _get_upload("image_b")
    if err:
        return err
    try:
        return jsonify(compare_images(data_a, data_b, normalize=_wants_normalize())), 200
    except Exception:
        app.logger.exception("Image comparison failed")
        return jsonify({"error": "Could not compare these photos. Please try different images."}), 500


@app.route("/api/compare-models", methods=["POST"])
def compare_models():
    data, err = _get_upload("image")
    if err:
        return err
    try:
        return jsonify(compare_classification_models(data, normalize=_wants_normalize())), 200
    except Exception:
        app.logger.exception("Model comparison failed")
        return jsonify({"error": "Could not compare the models on this photo. Please try a different image."}), 500


if __name__ == "__main__":
    print("\n  FreshSense (" + VERSION + ") is running.  Open this in your browser:  http://localhost:%d\n" % PORT)
    app.run(debug=False, host="0.0.0.0", port=PORT)
