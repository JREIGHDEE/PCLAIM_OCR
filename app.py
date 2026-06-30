from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from main import extract_text, extract_pdf_text
import cv2
import json
import numpy as np
import os
import tempfile
import time
from io import BytesIO
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
TRAINING_FOLDER = os.path.join(UPLOAD_FOLDER, "training")
REVIEWED_RESULTS_FOLDER = os.path.join(UPLOAD_FOLDER, "reviewed_results")
SESSIONS_FOLDER = os.path.join(REVIEWED_RESULTS_FOLDER, "sessions")
TRAINING_CATEGORIES = [
    "CASE #",
    "DATE & TIME OF ADMISSION",
    "NAME",
    "BDAY",
    "ADDRESS",
    "ADMITTING DIAGNOSIS",
    "DATE & TIME OF DELIVERY",
    "FINAL DIAGNOSIS",
    "DATE & TIME OF DISCHARGE"
]
CATEGORY_FOLDERS = {}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(REVIEWED_RESULTS_FOLDER, exist_ok=True)
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

for category in TRAINING_CATEGORIES:
    folder_name = secure_filename(category)
    folder_path = os.path.join(TRAINING_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    CATEGORY_FOLDERS[category] = folder_path

@app.route("/")
def home():
    return render_template(
        "index.html"
    )

@app.route("/ocr", methods=["POST"])
def run_ocr():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No file uploaded"
        })

    category = request.form.get("category", "").strip()
    if not category:
        return jsonify({
            "success": False,
            "error": "Please provide a category before OCR"
        }), 400

    file = request.files["image"]

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    try:

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension == ".pdf":

            tokens = extract_pdf_text(
                filepath
            )

        else:

            tokens = extract_text(
                filepath
            )

        enriched_tokens = [
            {
                **token,
                "category": category
            }
            for token in tokens
        ]

        raw_text = "\n".join(
            token["text"]
            for token in enriched_tokens
        )

        return jsonify({
            "success": True,
            "raw_text": raw_text,
            "category": category,
            "tokens": enriched_tokens
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


def get_case_session_path(case_id):
    safe_case_id = secure_filename(case_id) or "case"
    return os.path.join(SESSIONS_FOLDER, f"{safe_case_id}.json")


@app.route("/save_reviewed_result", methods=["POST"])
def save_reviewed_result():
    case_id = request.form.get("case_id", "").strip()
    if not case_id:
        return jsonify({
            "success": False,
            "error": "Please enter a case ID before saving"
        }), 400

    category = request.form.get("category", "").strip()
    reviewed_text = request.form.get("reviewed_text", "")
    review_values_raw = request.form.get("review_values", "")
    review_values = {}

    if review_values_raw:
        try:
            review_values = json.loads(review_values_raw)
            if not isinstance(review_values, dict):
                raise ValueError("review_values must be an object")
        except Exception:
            return jsonify({
                "success": False,
                "error": "Invalid review_values payload"
            }), 400

    if not review_values:
        if not category or category not in CATEGORY_FOLDERS:
            return jsonify({
                "success": False,
                "error": "Invalid or missing category"
            }), 400
        if not reviewed_text:
            return jsonify({
                "success": False,
                "error": "Please provide reviewed text before saving"
            }), 400

    case_name = request.form.get("case_name", "").strip()
    session_path = get_case_session_path(case_id)

    session_data = {}
    if os.path.exists(session_path):
        with open(session_path, "r", encoding="utf-8") as handle:
            try:
                session_data = json.load(handle)
            except Exception:
                session_data = {}

    if not isinstance(session_data, dict):
        session_data = {}

    session_data["case_id"] = case_id
    session_data["case_name"] = case_name or session_data.get("case_name", "")
    session_data["updated_at"] = int(time.time() * 1000)
    values = session_data.setdefault("values", {})
    if not isinstance(values, dict):
        values = {}
        session_data["values"] = values

    if review_values:
        for key, value in review_values.items():
            values[str(key).strip()] = str(value or "")
    else:
        values[category] = reviewed_text

    session_data["values"] = values

    with open(session_path, "w", encoding="utf-8") as handle:
        json.dump(session_data, handle, indent=2)

    return jsonify({
        "success": True,
        "saved_path": session_path
    })


@app.route("/detect_row_columns", methods=["POST"])
def detect_row_columns():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"detect_row_{int(time.time() * 1000)}.png"
    )

    try:
        file.save(temp_path)
        image = cv2.imread(temp_path)
        if image is None:
            raise Exception(f"Could not load image: {temp_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(3, image.shape[0] // 15))
        )
        vertical = cv2.morphologyEx(
            edged,
            cv2.MORPH_CLOSE,
            vertical_kernel,
            iterations=2
        )

        projection = np.sum(vertical, axis=0)
        threshold = max(1, int(np.max(projection) * 0.35))
        peaks = np.where(projection > threshold)[0]

        if len(peaks) < 2:
            return jsonify({
                "success": True,
                "columns": [0.33, 0.66]
            })

        columns = []
        last = -100
        min_gap = max(5, int(image.shape[1] * 0.02))
        for x in peaks:
            if x - last > min_gap:
                columns.append(x)
                last = x

        columns = [
            x for x in columns
            if x > image.shape[1] * 0.05 and x < image.shape[1] * 0.95
        ]

        if not columns:
            return jsonify({
                "success": True,
                "columns": [0.33, 0.66]
            })

        percents = [x / image.shape[1] for x in columns]
        return jsonify({
            "success": True,
            "columns": percents
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/export_case_session")
def export_case_session():
    case_id = request.args.get("case_id", "").strip()
    if not case_id:
        return jsonify({
            "success": False,
            "error": "Please enter a case ID before exporting"
        }), 400

    session_path = get_case_session_path(case_id)
    if not os.path.exists(session_path):
        return jsonify({
            "success": False,
            "error": "No saved session found for that case ID"
        }), 404

    with open(session_path, "r", encoding="utf-8") as handle:
        session_data = json.load(handle)

    values = session_data.get("values", {}) or {}
    row = {
        "Case ID": session_data.get("case_id", case_id),
        "Case Name": session_data.get("case_name", "")
    }
    for category in TRAINING_CATEGORIES:
        row[category] = values.get(category, "")

    extra_categories = [k for k in values.keys() if k not in TRAINING_CATEGORIES]
    for extra in sorted(extra_categories):
        row[extra] = values.get(extra, "")

    df = pd.DataFrame([row])

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"{secure_filename(case_id)}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/save_crop", methods=["POST"])
def save_cropped_field():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    category = request.form.get("category")
    if not category or category not in CATEGORY_FOLDERS:
        return jsonify({
            "success": False,
            "error": "Invalid or missing category"
        }), 400

    file = request.files["image"]
    filename = secure_filename(file.filename) or "crop.png"
    folder_path = CATEGORY_FOLDERS[category]
    timestamp = int(time.time() * 1000)
    save_path = os.path.join(folder_path, f"{timestamp}_{filename}")

    file.save(save_path)

    return jsonify({
        "success": True,
        "saved_path": save_path
    })


if __name__ == "__main__":
    app.run(debug=True)     