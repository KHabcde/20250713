"""
全ての.pyファイルをマージしたファイル
このファイルは自己参照を避けるため、マージ対象から除外されます。
"""

import os
import glob
import sys

def merge_python_files(output_filename="merged_files.py"):
    """
    フォルダ内の全ての.pyファイルをマージして一つのファイルに統合する
    
    Args:
        output_filename (str): 出力ファイル名
    """
    
    # 現在のスクリプトファイル名を取得（自己参照を避けるため）
    current_script = os.path.basename(__file__)
    output_file = output_filename
    
    # 現在のディレクトリの全ての.pyファイルを取得
    python_files = glob.glob("*.py")
    
    # 自分自身と出力ファイルを除外
    python_files = [f for f in python_files if f != current_script and f != output_file]
    
    # ファイルをアルファベット順にソート
    python_files.sort()
    
    print(f"マージ対象ファイル: {python_files}")
    print(f"出力ファイル: {output_file}")
    
    # マージされたコンテンツを格納するリスト
    merged_content = []
    
    # ヘッダーを追加
    merged_content.append("# " + "=" * 80)
    merged_content.append("# マージされたPythonファイル")
    merged_content.append("# 生成日時: " + str(__import__('datetime').datetime.now()))
    merged_content.append("# " + "=" * 80)
    merged_content.append("")
    
    # 各ファイルを処理
    for py_file in python_files:
        try:
            print(f"処理中: {py_file}")
            
            # ファイル区切りのヘッダーを追加
            merged_content.append("")
            merged_content.append("# " + "=" * 50)
            merged_content.append(f"# File: {py_file}")
            merged_content.append("# " + "=" * 50)
            merged_content.append("")
            
            # ファイルを読み込み
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # コンテンツを追加
            merged_content.append(content)
            merged_content.append("")
            
        except Exception as e:
            error_msg = f"エラー: {py_file} の読み込みに失敗しました - {str(e)}"
            print(error_msg)
            merged_content.append(f"# {error_msg}")
            merged_content.append("")
    
    # マージされたコンテンツをファイルに書き込み
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        print(f"\n✅ マージ完了! 出力ファイル: {output_file}")
        print(f"📁 マージされたファイル数: {len(python_files)}")
        print(f"📄 総行数: {len(merged_content)}")
        
    except Exception as e:
        print(f"❌ ファイルの書き込みに失敗しました: {str(e)}")

def show_file_info():
    """
    フォルダ内のPythonファイル情報を表示する
    """
    current_script = os.path.basename(__file__)
    python_files = glob.glob("*.py")
    python_files = [f for f in python_files if f != current_script]
    
    print("\n📋 フォルダ内のPythonファイル:")
    print("-" * 40)
    
    if not python_files:
        print("Pythonファイルが見つかりませんでした。")
        return
    
    total_lines = 0
    for py_file in sorted(python_files):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"📄 {py_file:<30} ({lines:>4} 行)")
        except Exception as e:
            print(f"❌ {py_file:<30} (読み込みエラー)")
    
    print("-" * 40)
    print(f"📊 総ファイル数: {len(python_files)}")
    print(f"📊 総行数: {total_lines}")

if __name__ == "__main__":
    print("🔧 Pythonファイルマージツール")
    print("=" * 50)
    
    # コマンドライン引数をチェック
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info" or sys.argv[1] == "-i":
            show_file_info()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
使用方法:
  python merged_python_files.py                    # デフォルトファイル名でマージ
  python merged_python_files.py output.py          # 指定ファイル名でマージ
  python merged_python_files.py --info             # ファイル情報を表示
  python merged_python_files.py --help             # このヘルプを表示
            """)
        else:
            # 出力ファイル名が指定された場合
            output_filename = sys.argv[1]
            merge_python_files(output_filename)
    else:
        # デフォルト動作
        show_file_info()
        print("\n🚀 マージを開始します...")
        merge_python_files()
