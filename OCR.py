### 事前にtesseractをインストールする必要がある
### 精度があまり良くない特に漢字,白背景黒文字ならなんとか
### ノイズ処理などが必要
### パラメータ調整：PSM,OEM
### カスタム言語モデルのトレーニングも可能
### PaddleOCRやEasyOCRの方が精度が良い、ただし遅いかも
### macであればocrmacが優秀？

import pytesseract
from paddleocr import PaddleOCR
import easyocr

class OCRTool:
    def __init__(self):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True)
        self.easy_ocr_reader = easyocr.Reader(['en'])

    def pytesseract_ocr(self, image_path):
        return pytesseract.image_to_string(image_path)

    def paddle_ocr_method(self, image_path):
        result = self.paddle_ocr.ocr(image_path, cls=True)
        text = [line[1][0] for line in result[0]]  # 最初の要素はテキスト行
        return '\n'.join(text)

    def easyocr_method(self, image_path):
        result = self.easy_ocr_reader.readtext(image_path)
        text = [text[1] for text in result]
        return '\n'.join(text)

# 使用例
ocr_tool = OCRTool()
print("Pytesseract OCR Result:")
print(ocr_tool.pytesseract_ocr('path_to_image.jpg'))

print("\nPaddleOCR Result:")
print(ocr_tool.paddle_ocr_method('path_to_image.jpg'))

print("\nEasyOCR Result:")
print(ocr_tool.easyocr_method('path_to_image.jpg'))
