# hook-mediapipe.py
from PyInstaller.utils.hooks import collect_all

# collect_all 会帮助你抓取 mediapipe 包里所有的
# - Python 源码 (datas)
# - 动态库文件 (binaries)
# - 隐藏导入 (hiddenimports)
datas, binaries, hiddenimports = collect_all('mediapipe')
