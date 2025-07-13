import os
import re

def split_merged_files(input_file: str = "merged_files.py", output_dir: str = "extracted_files"):
    """
    merged_python_files.pyで作成されたmerged_files.pyから各ファイルを抽出し、個別のファイルとして保存する。

    Parameters:
        input_file (str): merged_files.pyのパス（デフォルト: "merged_files.py"）
        output_dir (str): ファイルを出力するディレクトリ（デフォルト: "extracted_files"）
    """
    # 入力ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: 入力ファイル '{input_file}' が見つかりません")
        return
    
    # 出力先ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir}")

    # ファイル内容を読み込み
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return

    # 正規表現でファイルの区切りと内容を取得
    # merged_files.pyの形式: # File: ファイル名
    pattern = r"# File: (.+?)\n# =+\n\n(.*?)(?=\n\n# =+\n# File: |\Z)"
    matches = re.finditer(pattern, content, re.DOTALL)

    extracted_count = 0
    
    for match in matches:
        filename = match.group(1).strip()
        file_content = match.group(2).strip()
        
        # 空のファイル内容をスキップ
        if not file_content:
            print(f"⚠️  警告: '{filename}' は空の内容のためスキップします")
            continue
        
        # ファイル内容の最初と最後の余分な改行を除去
        file_content = file_content.strip()
        
        # 出力パスを作成
        output_path = os.path.join(output_dir, filename)
        
        try:
            # ファイルを書き出し
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(file_content + '\n')  # 最後に改行を追加
            
            print(f"✅ {filename} を保存しました ({len(file_content.splitlines())} 行)")
            extracted_count += 1
            
        except Exception as e:
            print(f"❌ ファイル書き込みエラー '{filename}': {e}")

    print(f"\n🎉 完了! {extracted_count} 個のファイルを抽出しました")
    
    if extracted_count == 0:
        print("⚠️  ファイルが見つかりませんでした。merged_files.pyの形式を確認してください。")

def show_merged_file_info(input_file: str = "merged_files.py"):
    """
    merged_files.py内に含まれるファイル情報を表示する
    
    Parameters:
        input_file (str): merged_files.pyのパス
    """
    if not os.path.exists(input_file):
        print(f"❌ エラー: 入力ファイル '{input_file}' が見つかりません")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return

    # ファイル名のみを抽出
    pattern = r"# File: (.+?)\n"
    matches = re.findall(pattern, content)
    
    print(f"\n📋 {input_file} 内のファイル一覧:")
    print("-" * 50)
    
    if not matches:
        print("ファイルが見つかりませんでした。")
        return
    
    for i, filename in enumerate(matches, 1):
        print(f"{i:2d}. {filename}")
    
    print("-" * 50)
    print(f"📊 総ファイル数: {len(matches)}")

# 使用例
if __name__ == "__main__":
    import sys
    
    print("🔧 Merged Files Extractor")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info" or sys.argv[1] == "-i":
            # ファイル情報を表示
            input_file = sys.argv[2] if len(sys.argv) > 2 else "merged_files.py"
            show_merged_file_info(input_file)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
使用方法:
  python split_text_to_files.py                    # デフォルト設定で抽出
  python split_text_to_files.py input.py out_dir   # 入力ファイルと出力ディレクトリを指定
  python split_text_to_files.py --info             # merged_files.py の情報を表示
  python split_text_to_files.py --info input.py    # 指定ファイルの情報を表示
  python split_text_to_files.py --help             # このヘルプを表示
            """)
        else:
            # 引数が指定された場合
            input_file = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_files"
            split_merged_files(input_file, output_dir)
    else:
        # デフォルト動作: まず情報を表示してから抽出
        show_merged_file_info()
        print("\n🚀 ファイル抽出を開始します...")
        split_merged_files()
