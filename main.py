from paddleocr import PaddleOCR
import cv2
import fitz
import tempfile
import os

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    show_log=False
)

def extract_text(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception(
            f"Could not load image: {image_path}"
        )

    result = ocr.ocr(
        img,
        cls=True
    )

    tokens = []

    if result and result[0]:

        for item in result[0]:

            text = item[1][0]
            conf = item[1][1]

            tokens.append({
                "text": text,
                "confidence": round(conf, 2)
            })

    return tokens


def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    all_tokens = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"ocr_page_{page_num}.png"
        )

        pix.save(temp_path)

        page_tokens = extract_text(
            temp_path
        )

        all_tokens.extend(
            page_tokens
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

    doc.close()

    return all_tokens


if __name__ == "__main__":

    path = "test.jpg"

    tokens = extract_text(path)

    for token in tokens:
        print(token["text"])