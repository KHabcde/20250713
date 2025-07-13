# ================================================================================
# マージされたPythonファイル
# 生成日時: 2025-07-13 15:49:02.542093
# ================================================================================


# ==================================================
# File: cmd_click.py
# ==================================================

# cmd = click
# 引数無し

import pyautogui

def cmd_click() :
    pyautogui.click()



# ==================================================
# File: cmd_cutrectanglefromwholess_nameofimage.py
# ==================================================

from PIL import Image
from rectangle_coordinate_strage import rectangle_storage

def cutrectanglefromwholess_nameofimage(arg1, arg2, arg3):
    """画像から指定された矩形領域を切り出して保存する

    Args:
        image_path (str): 元画像のファイルパス
        rectangle_name (str): 切り出す矩形領域の座標の名前
        output_name (str): 保存する画像のファイル名
    """
    image_path, rectangle_name, output_name = arg1, arg2, arg3
    
    try:
        # rectangle_storageから座標を取得
        coordinates = rectangle_storage.get(rectangle_name)
        if coordinates is None:
            print(f"エラー: 指定された名前の矩形領域 '{rectangle_name}' が見つかりません")
            return False
            
        # 画像を開く
        with Image.open(image_path) as img:
            # 座標を取得
            x1, y1, x2, y2 = coordinates
            
            # 画像を切り出す
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # 切り出した画像を保存
            cropped_img.save(output_name)
            print(f"切り出した画像を保存しました: {output_name}")
            return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False


# ==================================================
# File: cmd_diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate.py
# ==================================================

import cv2
import numpy as np
from PIL import Image
import os
from cmd_rectangle_coordinate_strage import cmd_rectangle_coordinate_strage

def cmd_diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate(arg1,arg2,arg3):
    """
    2枚のスクリーンショットから差分を検出し、その矩形領域の座標を指定した名前で保存する。

    Args:
        before_path (str): 1枚目の画像ファイルパス
        after_path (str): 2枚目の画像ファイルパス
        strage_name (str): 差分領域の座標を格納するタプルの名前
        target_x1 (int): 検出したい差分領域の左上X座標
        target_y1 (int): 検出したい差分領域の左上Y座標
        target_x2 (int): 検出したい差分領域の右下X座標
        target_y2 (int): 検出したい差分領域の右下Y座標
    """
    before_path, after_path, strage_name = arg1, arg2, arg3
    # 差分検出のパラメータ（最適な値を設定）
    threshold = 30          # 差分とみなすピクセルの輝度差しきい値
    min_area = 50          # 無視する最小領域（面積）
    hist_bins = 100        # 差分密度を分析する2Dヒストグラムの分割数
    focus_percentile = 90  # 密集度上位のビンを抽出するためのパーセンタイル
    padding = 50           # 最終的に出力する領域の上下左右に加える余白（ピクセル）

    # -------- 1. 画像の読み込みと形式変換 --------
    before_img = Image.open(before_path)
    after_img = Image.open(after_path)
    before_cv = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2BGR)
    after_cv = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2BGR)

    # -------- 2. 差分計算 --------
    diff = cv2.absdiff(before_cv, after_cv)  # 各ピクセルの差の絶対値
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # グレースケール化
    _, thresh_img = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)  # 2値化（差分強調）

    # -------- 3. 差分領域の抽出 --------
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:  # 面積フィルタでノイズ除去
            regions.append((x, y, w, h))

    if not regions:
        return  # 差分がなかった場合

    # -------- 4. 各差分領域の中心座標を取得 --------
    region_centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h in regions])

    # -------- 5. 2Dヒストグラムで密集度を分析 --------
    heatmap, xedges, yedges = np.histogram2d(
        region_centers[:, 0], region_centers[:, 1], bins=hist_bins
    )

    # -------- 6. ヒートマップの上位パーセンタイルを抽出 --------
    flattened = heatmap.flatten()
    threshold_value = np.percentile(flattened[flattened > 0], focus_percentile)
    dense_bins = np.where(heatmap >= threshold_value)

    if dense_bins[0].size == 0:
        return  # 密集領域がなかった場合

    # -------- 7. 高密度ビンを囲む最小矩形領域を計算 --------
    min_x_bin, max_x_bin = dense_bins[0].min(), dense_bins[0].max()
    min_y_bin, max_y_bin = dense_bins[1].min(), dense_bins[1].max()
    x1, x2 = int(xedges[min_x_bin]), int(xedges[max_x_bin + 1])
    y1, y2 = int(yedges[min_y_bin]), int(yedges[max_y_bin + 1])

    # -------- 8. パディングを追加して領域を拡張 --------
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(after_cv.shape[1], x2 + padding)
    y2 = min(after_cv.shape[0], y2 + padding)

    # -------- 9. 検出した差分領域の座標を保存 --------
    cmd_rectangle_coordinate_strage(strage_name, x1,y1, x2, y2)



# ==================================================
# File: cmd_doubleclick.py
# ==================================================

# cmd = doubleclick
# 引数無し

import pyautogui

def cmd_doubleclick() :
    pyautogui.doubleClick()



# ==================================================
# File: cmd_find_text_quarterofscreenshot_storagenameofcoordinate.py
# ==================================================

import cv2
import pytesseract
from typing import Optional, Tuple, List, Dict
import easyocr
import numpy as np
import difflib
import os
import json
import base64
from coordinate_strage import coordinate_strage
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- 設定 ---
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Tesseractの設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def cmd_find_text_quarterofscreenshot_storagenameofcoordinate(
    arg1: str,
    arg2: str,
    arg3: str,
    arg4: Optional[str] = None,
    arg5: Optional[str] = None,
    arg6: Optional[str] = None,
    arg7: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """
    スクリーンショットから指定されたテキストを探し、その座標を返す
    
    Args:
        text_to_find (str): 探したいテキスト
        coordinate_name (str): 座標に付ける名前
        screenshot_file (str): 読み込むスクリーンショットファイル
        x1 (int, optional): 検索領域の左上x座標
        y1 (int, optional): 検索領域の左上y座標
        x2 (int, optional): 検索領域の右下x座標
        y2 (int, optional): 検索領域の右下y座標
    
    Returns:
        Tuple[int, int] or None: テキストが見つかった場合は(x, y)座標、
                                見つからない場合はNone
    """
    try:
        # 引数を適切な変数に格納
        text_to_find = arg1
        coordinate_name = arg2
        screenshot_file = arg3
        x1 = arg4
        y1 = arg5
        x2 = arg6
        y2 = arg7

        def calculate_similarity(text1: str, text2: str) -> float:
            """テキスト間の類似度を計算"""
            return difflib.SequenceMatcher(None, text1, text2).ratio()

        # 座標値を整数に変換
        x1_int = int(x1) if x1 is not None else None
        y1_int = int(y1) if y1 is not None else None
        x2_int = int(x2) if x2 is not None else None
        y2_int = int(y2) if y2 is not None else None

        # スクリーンショット読み込み
        image = cv2.imread(screenshot_file)
        if image is None:
            raise FileNotFoundError(f"Screenshot file not found: {screenshot_file}")
            
        # 検索領域の切り出し
        if all([x1_int, y1_int, x2_int, y2_int]):
            image = image[y1_int:y2_int, x1_int:x2_int]

        # 画像の前処理をさらに強化
        scale_factor = 3  # スケールファクターを大きくする
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        resized_gray = cv2.resize(gray, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
        
        # ノイズ除去（強度を上げる）
        denoised = cv2.fastNlMeansDenoising(resized_gray, h=10)
        
        # コントラスト強調（より強く）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # シャープネス強調
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 二値化処理（適応的二値化を使用）
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        print(f"\n--- OCR結果の検証 ---")
        print(f"探索テキスト: '{text_to_find}'")

        # OCRの実行（Tesseractを使用）
        best_match = None
        highest_similarity = 0.6  # 類似度の閾値を下げて柔軟なマッチングを行う

        # Tesseractでの検出
        config = '--psm 11 --oem 3'
        ocr_output = pytesseract.image_to_data(binary, lang='jpn+eng', config=config, output_type=pytesseract.Output.DICT)
        
        for i in range(len(ocr_output['text'])):
            if ocr_output['conf'][i] > 30:  # 信頼度の閾値を30%に調整
                text = ocr_output['text'][i].strip()
                if text:
                    similarity = calculate_similarity(text, text_to_find)
                    confidence = ocr_output['conf'][i]
                    x = int(ocr_output['left'][i] / scale_factor)
                    y = int(ocr_output['top'][i] / scale_factor)
                    w = int(ocr_output['width'][i] / scale_factor)
                    h = int(ocr_output['height'][i] / scale_factor)
                    if x1_int is not None:
                        x += x1_int
                    if y1_int is not None:
                        y += y1_int
                    print(f"Tesseract - 検出テキスト: '{text}' (信頼度: {confidence}%, 類似度: {similarity:.2f}, 座標: ({x}, {y}), サイズ: {w}x{h})")
                    
                    # 完全一致の場合は即座に返す
                    if text.strip() == text_to_find:
                        x = int(ocr_output['left'][i] / scale_factor)
                        y = int(ocr_output['top'][i] / scale_factor)
                        if x1_int is not None:
                            x += x1_int
                        if y1_int is not None:
                            y += y1_int
                        print(f"\n✅ 完全一致が見つかりました:")
                        print(f"- テキスト: '{text}'")
                        print(f"- 座標: ({x}, {y})")
                        coordinate_strage.store(coordinate_name, x, y)
                        return (x, y)

                    # 類似度が閾値を超えた場合は候補として保存
                    elif similarity > highest_similarity:
                        highest_similarity = similarity
                        x = int(ocr_output['left'][i] / scale_factor)
                        y = int(ocr_output['top'][i] / scale_factor)
                        if x1_int is not None:
                            x += x1_int
                        if y1_int is not None:
                            y += y1_int
                        best_match = (x, y, text, "Tesseract")

        if best_match:
            x, y, matched_text, engine = best_match
            print(f"\n✅ 最適なマッチが見つかりました:")
            print(f"- テキスト: '{matched_text}'")
            print(f"- 座標: ({x}, {y})")
            print(f"- 類似度: {highest_similarity:.2f}")
            print(f"- 検出エンジン: {engine}")
            
            # 座標を保存
            coordinate_strage.store(coordinate_name, x, y)
            return (x, y)

        print(f"\nINFO: 完全一致/高類似度マッチが見つかりませんでした。LLMによる検証を試みます。")

        # OCR結果を収集
        ocr_results = []
        for i in range(len(ocr_output['text'])):
            if ocr_output['conf'][i] > 30:  # より多くの候補を含めるため閾値を下げる
                text = ocr_output['text'][i].strip()
                if text:
                    x = int(ocr_output['left'][i] / scale_factor)
                    y = int(ocr_output['top'][i] / scale_factor)
                    if x1_int is not None:
                        x += x1_int
                    if y1_int is not None:
                        y += y1_int
                    ocr_results.append({
                        "text": text,
                        "box": (x, y,
                               int(ocr_output['width'][i] / scale_factor),
                               int(ocr_output['height'][i] / scale_factor)),
                        "confidence": ocr_output['conf'][i] / 100.0
                    })

        # LLMによる検証
        ocr_candidates_str = "\n".join([f'- "{item["text"]}"' for item in ocr_results])
        prompt = f"""
        この画像を見てください。私は「{text_to_find}」というテキストを探しています。
        OCRでスキャンしたところ、以下のテキスト候補が見つかりました。
        --- OCR候補 ---
        {ocr_candidates_str}
        ---
        これらの候補の中から、私が探している「{text_to_find}」に意味的・視覚的に最も一致するものを特定してください。
        - 複数の候補が組み合わさって目的のテキストを形成している場合は、それら全てのテキストをリストで返してください。
        - 候補の中に完全一致するものがある場合も、そのテキストをリストで返してください。
        - 誤認識されているが文脈的に正しいと思われるものがあれば、そのOCRが読み取ったテキストを返してください。
        あなたの回答は、以下のJSON形式でお願いします。
        {{ "found": true, "texts": ["text1", "text2", ...] }}
        または、該当するものがなければ
        {{ "found": false, "texts": [] }}
        他の説明は不要です。
        """

        try:
            # LLMに問い合わせ
            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            print(f"DEBUG: LLMからの応答: {content}")
            result = json.loads(content)

            if result.get("found"):
                verified_texts = result["texts"]
                print(f"INFO: LLMが一致するテキストを特定しました: {verified_texts}")

                # テキストと座標の候補を収集
                text_candidates = []
                for item in ocr_results:
                    for vtext in verified_texts:
                        if vtext == item["text"]:  # 完全一致のみを考慮
                            text_candidates.append({
                                "text": item["text"],
                                "confidence": item["confidence"],
                                "box": item["box"]
                            })

                if text_candidates:
                    # x座標でソートして横並びの文字を検出
                    text_candidates.sort(key=lambda x: x["box"][0])
                    
                    # 連続した文字のグループを見つける
                    groups = [[text_candidates[0]]]
                    for i in range(1, len(text_candidates)):
                        prev = text_candidates[i-1]
                        curr = text_candidates[i]
                        # 横方向の距離が一定以内なら同じグループ
                        if curr["box"][0] - (prev["box"][0] + prev["box"][2]) < 30:
                            groups[-1].append(curr)
                        else:
                            groups.append([curr])

                    # 最も信頼度の高いグループを選択
                    best_group = max(groups, key=lambda g: sum(c["confidence"] for c in g))
                    
                    # 選択したグループの範囲を計算
                    min_x = min(c["box"][0] for c in best_group)
                    max_x = max(c["box"][0] + c["box"][2] for c in best_group)
                    min_y = min(c["box"][1] for c in best_group)
                    max_y = max(c["box"][1] + c["box"][3] for c in best_group)
                    
                    # 文字列の中心座標を計算
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2

                    print(f"\n✅ 複数のテキストの組み合わせが見つかりました:")
                    print(f"- テキスト: {verified_texts}")
                    print(f"- 検出された文字: {[c['text'] for c in best_group]}")
                    print(f"- 中心座標: ({center_x}, {center_y})")
                    print(f"- 文字領域: ({min_x}, {min_y}) - ({max_x}, {max_y})")

                    # 座標を保存して返す
                    coordinate_strage.store(coordinate_name, center_x, center_y)
                    return (center_x, center_y)

        except Exception as e:
            print(f"WARNING: LLMによる検証中にエラーが発生しました: {e}")

        print(f"\n❌ 適切なマッチが見つかりませんでした。")
        return None
        
    except Exception as e:
        print(f"Error in cmd_find_text_quarterofscreenshot_storagenameofcoordinate: {e}")
        return None



# ==================================================
# File: cmd_findtextfromcutimage_stragenameofcoordinate.py
# ==================================================

import cv2
import pytesseract
import Levenshtein
import easyocr
from coordinate_strage import coordinate_strage
from rectangle_coordinate_strage import rectangle_storage

# Tesseract OCRの設定
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# EasyOCRリーダーはグローバルで一度だけ初期化
EASYOCR_READER = None

def get_easyocr_reader():
    """EasyOCRリーダーをシングルトンとして取得"""
    global EASYOCR_READER
    if EASYOCR_READER is None:
        print("EasyOCRのリーダーを初回初期化しています...")
        EASYOCR_READER = easyocr.Reader(['ja', 'en'], gpu=False)
    return EASYOCR_READER

def preprocess_image(image):
    """画像をグレースケール化し、拡大・二値化して返す"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray_resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    _, binary_image = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_image = cv2.bitwise_not(binary_image)
    return inverted_image

def analyze_with_tesseract(image):
    """Tesseract OCRで画像を分析し、単語リストを返す"""
    config = '--psm 11 --oem 3'
    data = pytesseract.image_to_data(
        image, lang='jpn+eng', config=config, output_type=pytesseract.Output.DICT
    )
    
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip() != "":
            x = data['left'][i] // 2
            y = data['top'][i] // 2
            w = data['width'][i] // 2
            h = data['height'][i] // 2
            results.append({
                'text': data['text'][i],
                'conf': float(data['conf'][i]),
                'bbox': [x, y, x + w, y + h]
            })
    return results

def generate_candidate_phrases(ocr_results, max_gap_x=20, max_diff_y=10):
    """断片的なOCR結果から、連続する単語を結合して候補フレーズのリストを生成"""
    candidates = []
    if not ocr_results:
        return []

    ocr_results.sort(key=lambda r: r['bbox'][1])

    for i in range(len(ocr_results)):
        base_word = ocr_results[i]
        candidates.append(base_word.copy())
        
        x1, y1, x2, y2 = base_word['bbox']
        temp_line = [base_word]
        
        for j in range(i + 1, len(ocr_results)):
            next_word = ocr_results[j]
            is_same_line = abs((y1 + y2) / 2 - (next_word['bbox'][1] + next_word['bbox'][3]) / 2) < max_diff_y
            
            if is_same_line:
                temp_line.append(next_word)
        
        temp_line.sort(key=lambda r: r['bbox'][0])
        current_combined_word = temp_line[0]
        
        for k in range(1, len(temp_line)):
            next_word_in_line = temp_line[k]
            is_adjacent = (next_word_in_line['bbox'][0] - current_combined_word['bbox'][2]) < max_gap_x
            
            if is_adjacent:
                current_combined_word['text'] += next_word_in_line['text']
                current_combined_word['bbox'][2] = next_word_in_line['bbox'][2]
                current_combined_word['bbox'][1] = min(current_combined_word['bbox'][1], next_word_in_line['bbox'][1])
                current_combined_word['bbox'][3] = max(current_combined_word['bbox'][3], next_word_in_line['bbox'][3])
                current_combined_word['conf'] = min(current_combined_word['conf'], next_word_in_line['conf'])
                candidates.append(current_combined_word.copy())

    return candidates

def find_best_match(target_text, candidates):
    """候補リストの中から、ターゲットテキストに最も一致するものを探す"""
    if not candidates:
        return None

    best_candidate = None
    highest_score = -1.0

    for candidate in candidates:
        candidate_text = candidate['text'].replace(" ", "")
        distance = Levenshtein.distance(target_text, candidate_text)
        longer_len = max(len(target_text), len(candidate_text))
        
        if longer_len == 0:
            similarity = 1.0 if distance == 0 else 0.0
        else:
            similarity = 1.0 - (distance / longer_len)
        
        length_ratio = min(len(target_text), len(candidate_text)) / max(len(target_text), len(candidate_text)) if longer_len > 0 else 0
        score = similarity * 0.8 + length_ratio * 0.2

        if score > highest_score:
            highest_score = score
            best_candidate = candidate
            best_candidate['score'] = score
    
    return best_candidate

def convert_to_global_coords(local_bbox, crop_origin):
    """局所座標をスクリーンショット全体の絶対座標に変換"""
    local_x1, local_y1, local_x2, local_y2 = local_bbox
    crop_x, crop_y = crop_origin
    return [local_x1 + crop_x, local_y1 + crop_y, local_x2 + crop_x, local_y2 + crop_y]

def find_text_coordinates(cropped_image_path: str, crop_area_vertices: tuple, target_text: str):
    """画像とターゲットテキストから、そのテキストの全体座標を推定"""
    print(f"--- ターゲット「{target_text}」の座標推定を開始 ---")

    cropped_image = cv2.imread(cropped_image_path)
    if cropped_image is None:
        print(f"エラー: 画像 '{cropped_image_path}' が読み込めません。")
        return None
    
    processed_image = preprocess_image(cropped_image)
    print("OCRを実行中...")
    ocr_results = analyze_with_tesseract(processed_image)
    
    if not ocr_results:
        print("OCRでテキストが検出されませんでした。")
        return None

    print("候補フレーズを生成中...")
    candidates = generate_candidate_phrases(ocr_results)
    
    print("最適な候補を検索中...")
    best_match = find_best_match(target_text, candidates)
    
    SCORE_THRESHOLD = 0.7
    
    if best_match and best_match['score'] > SCORE_THRESHOLD:
        print("\n[成功] ターゲットに一致する候補が見つかりました！")
        print(f"  - 検出テキスト: '{best_match['text']}'")
        print(f"  - マッチスコア: {best_match['score']:.2f}")

        crop_origin = (crop_area_vertices[0], crop_area_vertices[1])
        global_bbox = convert_to_global_coords(best_match['bbox'], crop_origin)
        
        center_x = global_bbox[0] + (global_bbox[2] - global_bbox[0]) // 2
        center_y = global_bbox[1] + (global_bbox[3] - global_bbox[1]) // 2

        result = {
            "center": {"x": center_x, "y": center_y},
            "bbox": global_bbox,
            "text": best_match['text'],
            "score": best_match['score']
        }
        return result
    else:
        print("\n[失敗] ターゲットに一致する信頼性の高い候補が見つかりませんでした。")
        if best_match:
            print(f"  (最も近かった候補: '{best_match['text']}', スコア: {best_match['score']:.2f})")
        return None

def cmd_findtextfromcutimage_stragenameofcoordinate(target_text: str, image_path: str, rectangle_name: str, coordinate_name: str):
    """
    切り出し画像から特定のテキストを探し、その座標を保存する
    
    Args:
        target_text (str): 探したいテキスト
        image_path (str): 切り出し画像のファイルパス
        rectangle_name (str): 切り出し画像の領域情報が格納されているタプルの名前
        coordinate_name (str): 特定した座標を格納するタプルの名前
    """
    # 切り出し領域の座標を取得
    crop_area = rectangle_storage.get(rectangle_name)
    if crop_area is None:
        print(f"エラー: 指定された矩形領域 '{rectangle_name}' が見つかりません。")
        return

    # テキストの座標を探索
    result = find_text_coordinates(image_path, crop_area, target_text)
    if result is None:
        print(f"エラー: テキスト '{target_text}' の座標を特定できませんでした。")
        return

    # 特定した座標を保存
    coordinate_strage.store(coordinate_name, result["center"]["x"], result["center"]["y"])
    print(f"座標を '{coordinate_name}' として保存しました: ({result['center']['x']}, {result['center']['y']})")



# ==================================================
# File: cmd_move_relative.py
# ==================================================

# cmd = cmd_move_relative
# arg1 = +x 
# arg2 = +y

import pyautogui

def cmd_move_relative(arg1, arg2):
    if arg1 is not None and arg2 is not None:
        pyautogui.move(int(arg1), int(arg2))
    else:
        print("Error: 'move_relative' command requires two arguments (dx, dy).")



# ==================================================
# File: cmd_move_to_coordinateofthepoint.py
# ==================================================

# cmd = cmd_move_to_coordinateofthepoint
# arg1 = x 
# arg2 = y

import pyautogui

def cmd_move_to_coordinateofthepoint(arg1, arg2):
    if arg1 is not None and arg2 is not None:
        pyautogui.moveTo(int(arg1), int(arg2))
    else:
        print("Error: 'move_to_coordinateofthepoint' command requires two arguments (x, y).")



# ==================================================
# File: cmd_move_to_storedcoordinate.py
# ==================================================

import pyautogui
from typing import Optional, Tuple
from coordinate_strage import coordinate_strage

def cmd_move_to_storedcoordinate(arg1) -> bool:
    """
    coordinate_strageに保存された座標位置までカーソルを移動する
    
    Args:
        coordinate_name (str): カーソルを移動したい座標の名前（例：coordinate_1）
    
    Returns:
        bool: カーソル移動の成功(True)・失敗(False)
    """
    coordinate_name = arg1
    
    try:
        # coordinate_strageから座標を取得
        coordinates = coordinate_strage.get(coordinate_name)
        
        if coordinates is None:
            print(f"Error: Coordinates not found for name: {coordinate_name}")
            return False
            
        # 座標を取り出す
        x, y = coordinates
        
        # カーソル移動実行
        pyautogui.moveTo(x, y)
        return True
        
    except Exception as e:
        print(f"Error in cmd_click_stored: {e}")
        return False



# ==================================================
# File: cmd_rectangle_coordinate_strage.py
# ==================================================

from rectangle_coordinate_strage import rectangle_storage

def cmd_rectangle_coordinate_strage(name: str, top_left_x: str, top_left_y: str, bottom_right_x: str, bottom_right_y: str) -> None:
    """
    矩形領域の頂点座標を登録する
    
    Args:
        name (str): 矩形領域の名前
        top_left_x (str): 左上のX座標（文字列で受け取り、intに変換）
        top_left_y (str): 左上のY座標（文字列で受け取り、intに変換）
        bottom_right_x (str): 右下のX座標（文字列で受け取り、intに変換）
        bottom_right_y (str): 右下のY座標（文字列で受け取り、intに変換）
    """
    try:
        # 文字列で渡される座標をintに変換
        tlx = int(top_left_x)
        tly = int(top_left_y)
        brx = int(bottom_right_x)
        bry = int(bottom_right_y)
        
        # rectangle_storageに座標を保存
        rectangle_storage.store(name, tlx, tly, brx, bry)
        print(f"矩形領域 '{name}' を登録しました: 左上({tlx}, {tly}), 右下({brx}, {bry})")
        
    except ValueError:
        print("エラー: 座標は数値である必要があります")
    except Exception as e:
        print(f"エラー: 矩形領域の登録に失敗しました - {str(e)}")



# ==================================================
# File: cmd_screenshot.py
# ==================================================

# cmd = cmd_screenshot
# arg1 = 保存するファイル名

import pyautogui

def cmd_screenshot(arg1: str):
    if arg1 is not None:
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(arg1)
            print(f"Screenshot saved as: {arg1}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    else:
        print("Error: 'screenshot' command requires one argument (filename).")



# ==================================================
# File: cmd_sleep.py
# ==================================================

# cmd = cmd_sleep
# arg1 = 秒数
import pyautogui

def cmd_sleep(arg1):
    if arg1 is not None:
        try:
            pyautogui.sleep(float(arg1))
        except ValueError:
            print("Error: 'sleep' command requires a numeric argument (seconds).")
    else:
        print("Error: 'sleep' command requires one argument (seconds).")



# ==================================================
# File: cmd_templatematch_stragecoordinatenameofthepoint.py
# ==================================================

import cv2
import numpy as np
import os
from typing import Optional

# 1で作成した coordinate_storage.py から coordinate_strage インスタンスをインポート
from coordinate_strage import coordinate_strage

def cmd_templatematch_stragecoordinatenameofthepoint(arg1,arg2,arg3) -> None:
    screenshot_path,template_path,coordinate_name = arg1, arg2, arg3
    """
    スクリーンショット画像からテンプレート画像を検索し、最も一致する領域の中心座標を指定された名前で保存する。

    Args:
        screenshot_path (str): 検索対象となるスクリーンショット画像のファイルパス (arg1)。
        template_path (str): 検索するテンプレート画像のファイルパス (arg2)。
        coordinate_name (str): 見つかった座標を保存する際の名前 (arg3)。
    """
    # --- 引数のチェック ---
    if not all([screenshot_path, template_path, coordinate_name]):
        print("エラー: 必要な引数が不足しています (スクリーンショットパス, テンプレートパス, 座標名)。")
        return

    # --- ファイル存在チェック ---
    if not os.path.exists(screenshot_path):
        print(f"エラー: スクリーンショットファイルが見つかりません: {screenshot_path}")
        return
    if not os.path.exists(template_path):
        print(f"エラー: テンプレートファイルが見つかりません: {template_path}")
        return

    try:
        # --- 画像の読み込み ---
        # 画像をそのままの色で読み込む
        screenshot_img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

        if screenshot_img is None:
            print(f"エラー: スクリーンショット画像の読み込みに失敗しました: {screenshot_path}")
            return
        if template_img is None:
            print(f"エラー: テンプレート画像の読み込みに失敗しました: {template_path}")
            return

        # テンプレート画像の幅と高さを取得
        h, w = template_img.shape[:2]

        # --- テンプレートマッチングの実行 ---
        # TM_CCOEFF_NORMED: 類似度を-1から1の範囲で正規化して評価する手法。1に近いほど一致。
        result = cv2.matchTemplate(screenshot_img, template_img, cv2.TM_CCOEFF_NORMED)

        # --- 最も一致する位置の特定 ---
        # resultの中から最もスコアが高い(一致度が高い)場所を探す
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # オプション: 一致スコアが低い場合は警告を出すことも可能
        if max_val < 0.8: # この閾値は必要に応じて調整してください
            print(f"警告: マッチングスコアが低いです ({max_val:.4f})。意図しない場所を検出した可能性があります。")

        # --- 中心座標の計算 ---
        top_left = max_loc
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        
        # TM_CCOEFF_NORMEDを使用した場合、最もスコアが高い位置(max_loc)が、
        # 最も一致する領域の中心の座標となる
        print(f"テンプレートマッチング結果: スコア={max_val:.4f}, 中心座標={center_x}, {center_y} ")

        # --- 座標の保存 ---
        # coordinate_storageモジュールのグローバルインスタンスを使用して座標を保存
        coordinate_strage.store(coordinate_name, center_x, center_y)

    except cv2.error as e:
        print(f"OpenCVエラーが発生しました: {e}")
        # テンプレート画像がスクリーンショット画像より大きい場合にこのエラーが発生しやすい
        if "template is larger than image" in str(e):
             print("詳細: テンプレート画像がスクリーンショット画像よりも大きいです。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# ==================================================
# File: cmd_templatematch_stragecoordinatenameoftherectangle.py
# ==================================================

import cv2
import numpy as np
import os
from typing import Optional, Tuple

# rectangle_coordinate_strage.py から rectangle_storage インスタンスをインポート
from rectangle_coordinate_strage import rectangle_storage

def cmd_templatematch_stragecoordinatenameoftherectangle(arg1, arg2, arg3) -> None:
    screenshot_path, template_path, rectangle_name = arg1, arg2, arg3
    """
    スクリーンショット画像からテンプレート画像を検索し、最も一致する領域の矩形座標を指定された名前で保存する。

    Args:
        screenshot_path (str): 検索対象となるスクリーンショット画像のファイルパス (arg1)。
        template_path (str): 検索するテンプレート画像のファイルパス (arg2)。
        rectangle_name (str): 見つかった矩形領域を保存する際の名前 (arg3)。
    """
    # --- 引数のチェック ---
    if not all([screenshot_path, template_path, rectangle_name]):
        print("エラー: 必要な引数が不足しています (スクリーンショットパス, テンプレートパス, 矩形領域名)。")
        return

    # --- ファイル存在チェック ---
    if not os.path.exists(screenshot_path):
        print(f"エラー: スクリーンショットファイルが見つかりません: {screenshot_path}")
        return
    if not os.path.exists(template_path):
        print(f"エラー: テンプレートファイルが見つかりません: {template_path}")
        return

    try:
        # --- 画像の読み込み ---
        # 画像をそのままの色で読み込む
        screenshot_img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

        if screenshot_img is None:
            print(f"エラー: スクリーンショット画像の読み込みに失敗しました: {screenshot_path}")
            return
        if template_img is None:
            print(f"エラー: テンプレート画像の読み込みに失敗しました: {template_path}")
            return

        # テンプレート画像の幅と高さを取得
        h, w = template_img.shape[:2]

        # --- テンプレートマッチングの実行 ---
        # TM_CCOEFF_NORMED: 類似度を-1から1の範囲で正規化して評価する手法。1に近いほど一致。
        result = cv2.matchTemplate(screenshot_img, template_img, cv2.TM_CCOEFF_NORMED)

        # --- 最も一致する位置の特定 ---
        # resultの中から最もスコアが高い(一致度が高い)場所を探す
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # オプション: 一致スコアが低い場合は警告を出すことも可能
        if max_val < 0.8: # この閾値は必要に応じて調整してください
            print(f"警告: マッチングスコアが低いです ({max_val:.4f})。意図しない場所を検出した可能性があります。")

        # --- 矩形領域の座標計算 ---
        top_left_x = max_loc[0]
        top_left_y = max_loc[1]
        bottom_right_x = top_left_x + w
        bottom_right_y = top_left_y + h
        
        # 矩形領域の座標をログに出力
        print(f"テンプレートマッチング結果: スコア={max_val:.4f}")
        print(f"矩形領域: 左上=({top_left_x}, {top_left_y}), 右下=({bottom_right_x}, {bottom_right_y})")

        # --- 矩形領域の座標を保存 ---
        # rectangle_storageモジュールのグローバルインスタンスを使用して座標を保存
        rectangle_storage.store(rectangle_name, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        print(f"矩形領域の座標を '{rectangle_name}' として保存しました")

    except cv2.error as e:
        print(f"OpenCVエラーが発生しました: {e}")
        # テンプレート画像がスクリーンショット画像より大きい場合にこのエラーが発生しやすい
        if "template is larger than image" in str(e):
             print("詳細: テンプレート画像がスクリーンショット画像よりも大きいです。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# ==================================================
# File: cmd_type.py
# ==================================================

# cmd = cmd_type
# arg1 = 文字列

import pyautogui

def cmd_type(arg1):
    if arg1 is not None:
        pyautogui.typewrite(arg1)
    else:
        print("Error: 'type' command requires one argument (text to type).")



# ==================================================
# File: cmdtest.py
# ==================================================

from interpreter import execute_macro
from parser_and_supplementer import macro_parse_and_supplement
import os

def test_macro_execution(macro: str) -> None:
    """
    マクロを直接実行してテストする関数
    
    Args:
        macro (str): 実行したいマクロ命令
    """
    print(f"実行するマクロ: {macro}")
    
    # マクロをパースして補完
    parsed_macro = macro_parse_and_supplement(macro)
    print(f"パース結果: {parsed_macro}")
    
    # マクロを実行
    execute_macro(parsed_macro)
    print("実行完了\n" + "-"*50 + "\n")

def main():
    # # テストケース0: 矩形領域の頂点を登録
    # test_macro_execution(
    #     "diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate before_パワーポイント.png after_パワーポイント.png rectangle1"
    # )
    
    # # テストケース1: 登録した矩形領域を切り出して保存
    # test_macro_execution(
    #     "cutrectanglefromwholess_nameofimage after_パワーポイント.png rectangle1 cropped_image_パワーポイント.png"
    # )
    
    # # テストケース2: 保存した座標に移動
    test_macro_execution(
        "templatematch_stragecoordinatenameofthepoint screenshot.png button.png button_pos"
    )
    
    # テストケース3: クリック
    test_macro_execution(
        "move_to_storedcoordinate button_pos"
    )
    
    # テストケース4: 矩形領域の座標を登録
    test_macro_execution(
        "templatematch_stragecoordinatenameoftherectangle screenshot.png button.png button_rect"
    )

if __name__ == "__main__":
    # テスト実行前の確認
    main()



# ==================================================
# File: coordinate_strage.py
# ==================================================

from typing import Dict, Tuple, Optional

class CoordinateStorage:
    def __init__(self):
        self._storage: Dict[str, Tuple[int, int]] = {}

    def store(self, name: str, x: int, y: int) -> None:
        """
        座標を保存する
        
        Args:
            name (str): 座標の名前
            x (int): X座標
            y (int): Y座標
        """
        self._storage[name] = (x, y)

    def get(self, name: str) -> Optional[Tuple[int, int]]:
        """
        保存された座標を取得する
        
        Args:
            name (str): 座標の名前
            
        Returns:
            Tuple[int, int] or None: 保存された(x, y)座標。存在しない場合はNone
        """
        return self._storage.get(name)

    def remove(self, name: str) -> bool:
        """
        保存された座標を削除する
        
        Args:
            name (str): 座標の名前
            
        Returns:
            bool: 削除に成功した場合はTrue、座標が存在しなかった場合はFalse
        """
        if name in self._storage:
            del self._storage[name]
            return True
        return False

    def clear(self) -> None:
        """全ての保存された座標を削除する"""
        self._storage.clear()

    def list_coordinates(self) -> Dict[str, Tuple[int, int]]:
        """
        全ての保存された座標を返す
        
        Returns:
            Dict[str, Tuple[int, int]]: 名前と座標のペアを含む辞書
        """
        return self._storage.copy()

# グローバルなインスタンスを作成
coordinate_strage = CoordinateStorage()



# ==================================================
# File: interpreter.py
# ==================================================

import pyautogui
import cv2
import numpy as np
from PIL import Image
import pytesseract
import difflib
from typing import Optional, Tuple, List, Dict
#各コマンドの関数をインポート
from cmd_click import cmd_click
from cmd_doubleclick import cmd_doubleclick
from cmd_move_to_coordinateofthepoint import cmd_move_to_coordinateofthepoint
from cmd_move_relative import cmd_move_relative
from cmd_sleep import cmd_sleep
from cmd_type import cmd_type
from cmd_screenshot import cmd_screenshot
# from cmd_find_text_quarterofscreenshot_storagenameofcoordinate import cmd_find_text_quarterofscreenshot_storagenameofcoordinate
from cmd_move_to_storedcoordinate import cmd_move_to_storedcoordinate
# from cmd_findtextfromcutimage_stragenameofcoordinate import cmd_findtextfromcutimage_stragenameofcoordinate
from cmd_rectangle_coordinate_strage import cmd_rectangle_coordinate_strage
from cmd_diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate import cmd_diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate
from cmd_cutrectanglefromwholess_nameofimage import cutrectanglefromwholess_nameofimage
from cmd_templatematch_stragecoordinatenameofthepoint import cmd_templatematch_stragecoordinatenameofthepoint
from cmd_templatematch_stragecoordinatenameoftherectangle import cmd_templatematch_stragecoordinatenameoftherectangle

# Tesseractの実行ファイルのパスを設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    from pywinauto import Application
except ImportError:
    print(" pywinauto is not installed. Some functionalities will be unavailable.")

def execute_macro(parsed_macro: dict):

    """
    Executes a parsed macro command using PyAutoGUI.

    Args:
        parsed_macro (dict): A dictionary containing the parsed macro components.
    """
    cmd = parsed_macro.get('cmd')
    arg1 = parsed_macro.get('arg1')
    arg2 = parsed_macro.get('arg2')
    arg3 = parsed_macro.get('arg3')
    arg4 = parsed_macro.get('arg4')
    arg5 = parsed_macro.get('arg5')
    arg6 = parsed_macro.get('arg6')
    arg7 = parsed_macro.get('arg7')
    arg8 = parsed_macro.get('arg8')


    if cmd == 'move_to_coordinateofthepoint':
        cmd_move_to_coordinateofthepoint(arg1, arg2)
    elif cmd == 'move_relative':
        cmd_move_relative(arg1, arg2)
    elif cmd == 'click':
        cmd_click()
    elif cmd == 'doubleclick':
        cmd_doubleclick()
    elif cmd == 'sleep':
        cmd_sleep(arg1)
    elif cmd == 'type':
        cmd_type(arg1)
    elif cmd == 'screenshot':
        cmd_screenshot(arg1)
    # elif cmd == 'find_text_quarterofscreenshot_storagenameofcoordinate':
    #     cmd_find_text_quarterofscreenshot_storagenameofcoordinate(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    elif cmd == 'move_to_storedcoordinate':
        cmd_move_to_storedcoordinate(arg1)
    # elif cmd == 'findtextfromcutimage_stragenameofcoordinate':
    #     cmd_findtextfromcutimage_stragenameofcoordinate(arg1, arg2, arg3, arg4)
    # elif cmd == 'rectangle_coordinate_strage':
    #     cmd_rectangle_coordinate_strage(arg1, arg2, arg3, arg4, arg5)
    elif cmd == 'diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate':
        cmd_diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate(arg1, arg2, arg3)
    elif cmd == 'rectangle_coordinate_strage':
        cmd_rectangle_coordinate_strage(arg1, arg2, arg3, arg4)
    elif cmd == 'cutrectanglefromwholess_nameofimage':
        cutrectanglefromwholess_nameofimage(arg1, arg2, arg3)
    elif cmd == 'templatematch_stragecoordinatenameofthepoint':
        cmd_templatematch_stragecoordinatenameofthepoint(arg1, arg2, arg3)
    elif cmd == 'templatematch_stragecoordinatenameoftherectangle':
        cmd_templatematch_stragecoordinatenameoftherectangle(arg1, arg2, arg3)


# ==================================================
# File: main.py
# ==================================================

==================================================

from transpiler import natural_to_macro
from interpreter import execute_macro
from parser_and_supplementer import macro_parse_and_supplement

def process_natural_language(file_path):
    """
    Reads natural language instructions from a file, converts them to macro syntax,
    and executes the macros.

    Args:
        file_path (str): Path to the file containing natural language instructions.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        natural_command = line.strip()
        if not natural_command:
            continue

        print(f"Processing: {natural_command}")

        # Convert natural language to macro
        macro = natural_to_macro(natural_command)
        print(f"Converted Macro: {macro}")

        # Parse and supplement the macro
        parsed_macro = macro_parse_and_supplement(macro)
        print(f"Parsed Macro: {parsed_macro}")

        # Execute the parsed macro
        execute_macro(parsed_macro)

if __name__ == "__main__":
    process_natural_language("natural_text.txt")


==================================================


# ==================================================
# File: parser_and_supplementer.py
# ==================================================



def macro_parse_and_supplement(macro):
    """
    Parses the macro and supplements it with additional logic if necessary.

    Args:
        macro (str): The macro string to parse and supplement.

    Returns:
        dict: A dictionary containing the parsed macro and any additional data.
    """
    parts = macro.split(" ")
    parsed_macro = {
        "cmd": parts[0],
        "arg1": parts[1] if len(parts) > 1 else None,
        "arg2": parts[2] if len(parts) > 2 else None,
        "arg3": parts[3] if len(parts) > 3 else None,
        "arg4": parts[4] if len(parts) > 4 else None,
        "arg5": parts[5] if len(parts) > 5 else None,
        "arg6": parts[6] if len(parts) > 6 else None,
        "arg7": parts[7] if len(parts) > 7 else None,
    }

    return parsed_macro


# ==================================================
# File: rectangle_coordinate_strage.py
# ==================================================

from typing import Dict, Tuple, Optional

class RectangleStorage:
    def __init__(self):
        self._storage: Dict[str, Tuple[int, int, int, int]] = {}

    def store(self, name: str, top_left_x: int, top_left_y: int, bottom_right_x: int, bottom_right_y: int) -> None:
        """
        矩形領域の座標を保存する
        
        Args:
            name (str): 矩形領域の名前
            top_left_x (int): 左上のX座標
            top_left_y (int): 左上のY座標
            bottom_right_x (int): 右下のX座標
            bottom_right_y (int): 右下のY座標
        """
        self._storage[name] = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def get(self, name: str) -> Optional[Tuple[int, int, int, int]]:
        """
        保存された矩形領域の座標を取得する
        
        Args:
            name (str): 矩形領域の名前
            
        Returns:
            Tuple[int, int, int, int] or None: 保存された(左上x, 左上y, 右下x, 右下y)座標。存在しない場合はNone
        """
        return self._storage.get(name)

    def remove(self, name: str) -> bool:
        """
        保存された矩形領域の座標を削除する
        
        Args:
            name (str): 矩形領域の名前
            
        Returns:
            bool: 削除に成功した場合はTrue、座標が存在しなかった場合はFalse
        """
        if name in self._storage:
            del self._storage[name]
            return True
        return False

    def clear(self) -> None:
        """全ての保存された矩形領域の座標を削除する"""
        self._storage.clear()

    def list_rectangles(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        全ての保存された矩形領域の座標を返す
        
        Returns:
            Dict[str, Tuple[int, int, int, int]]: 名前と矩形領域の座標のペアを含む辞書
        """
        return self._storage.copy()

# グローバルなインスタンスを作成
rectangle_storage = RectangleStorage()



# ==================================================
# File: split_text_to_files.py
# ==================================================

import os
import re

def split_text_to_files(input_file: str, output_dir: str = "output_files"):
    """
    指定されたテキストファイルから 'File: ファイル名' を検出し、それぞれのファイルに内容を書き出す。

    Parameters:
        input_file (str): 入力テキストファイルのパス
        output_dir (str): ファイルを出力するディレクトリ
    """
    # 出力先ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正規表現でファイルの区切りと内容を取得
    pattern = r"^File: (.+?)\n(.*?)(?=^File: |\Z)"  # 次の File: または文末まで
    matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)

    for match in matches:
        filename = match.group(1).strip()
        file_content = match.group(2).strip()

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        print(f"✅ {filename} を保存しました")

# 使用例
if __name__ == "__main__":
    split_text_to_files("all_files.txt")  # 元のテキストを all_files.txt と仮定



# ==================================================
# File: transpiler.py
# ==================================================

#from openai import OpenAI
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from pywinauto.application import Application

# Load environment variables
load_dotenv()

# Get the OpenAI API key
#api_key = os.getenv("OPENAI_API_KEY")
#if not api_key:
#    raise ValueError("OPENAI_API_KEY が設定されていません")
#
#client = OpenAI(api_key=api_key)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def natural_to_macro(text: str) -> str:
    """
    Converts natural language text to macro syntax using OpenAI's ChatGPT API and appends the result to macro_syntax.txt.

    Args:
        text (str): The natural language command.

    Returns:
        str: The converted macro syntax.
    """
    try:
        # プロンプトテンプレートは直接文字列として定義
        # Format the template with the input text
        prompt = f"""
        以下の自然言語指示を、対応するマクロ構文に変換してください。
        自然言語指示: "{text}"
        マクロ構文のみを出力してください。
        出力には余計な記号や装飾を含めないでください。
        
        # 自然言語の例:
        # 「マウスカーソルを(100,200)に移動して」
        # マクロ構文の例:
        # move_to_coordinateofthepoint 100 200

        # 自然言語の例:
        # マウスカーソルをX方向に100,Y方向に200移動して
        # マクロ構文の例:
        # move_relative 100 200
        
        # 自然言語の例:
        # クリックして
        # マクロ構文の例:
        # click
        
        # 自然言語の例:
        # ダブルクリックして
        # マクロ構文の例:
        # doubleclick
        
        # 自然言語の例:
        # 0.5秒待機して
        # マクロ構文の例:
        # sleep 0.5
        
        # 自然言語の例:
        # 「c」とタイプしてください
        # マクロ構文の例:
        # type c
        
        # 自然言語の例:
        # スクリーンショットを撮り、「before.png」という名前で保存してください
        # マクロ構文の例:
        # screenshot before.png
        
        # 自然言語の例:
        # 「before.png」という画像の、「左上」に注目し、「ファイル」というテキストの位置を特定し、「coordinate_1」という名前で保存してください
        # マクロ構文の例:
        # find_text_quarterofscreenshot_storagenameofcoordinate ファイル coordinate_1 before.png 0 0 960 540
        
        # 自然言語の例:
        # マウスカーソルを登録済みの座標「coordinate_1」に移動して
        # マクロ構文の例:
        # move_to_storedcoordinate coordinate_1
        
        # 自然言語の例:
        # 「cut_image.png」の中から、登録済みの矩形領域「rect1」内で「メニュー」というテキストを探し、その位置を「menu_pos」という名前で保存して
        # マクロ構文の例:
        # findtextfromcutimage_stragenameofcoordinate メニュー cut_image.png rect1 menu_pos

        # 自然言語の例:
        # 左上が(100, 200)で右下が(300, 400)の矩形領域を「search_area」という名前で登録して
        # マクロ構文の例:
        # rectangle_coordinate_strage search_area 100 200 300 400
        
        # 自然言語の例:
        # 「before.png」と「after.png」の2枚のスクリーンショットから差分を検出し、その矩形領域の座標を「rectangle1」という名前で保存して
        # マクロ構文の例:
        # diffdetectwithparameter_betweenbeforessandafterss_stragenameofrectanglecoordinate before.png after.png rectangle1
        
        # 自然言語の例:
        # 「before.png」の全体から、「rectangle1」の矩形領域を切り出して「rectangle1_image.png」という名前で保存して
        # マクロ構文の例:
        # cutrectanglefromwholess_nameofimage before.png rectangle1 rectangle1_image.png
        
        # 自然言語の例:
        # 「screenshot.png」の中から「button.png」という画像を探し、その中心座標を「button_pos」という名前で保存して
        # マクロ構文の例:
        # templatematch_stragecoordinatenameofthepoint screenshot.png button.png button_pos
        
        # 自然言語の例:
        # スクリーンショット「screenshot.png」の中から「button.png」という画像を探し、その矩形領域を「button_rect」という名前で保存して
        # マクロ構文の例:
        # templatematch_stragecoordinatenameoftherectangle screenshot.png button.png button_rect
        """
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
            )
        if not response.choices or not response.choices[0].message.content.strip():
            raise ValueError("Empty response from the API.")

        macro_syntax = response.choices[0].message.content.strip()

        # Append the macro syntax to macro_syntax.txt
        with open("macro_syntax.txt", "a", encoding="utf-8") as output_file:
            output_file.write(macro_syntax + "\n")

        return macro_syntax
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "Error: Empty response from the API."
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error: API call failed."
