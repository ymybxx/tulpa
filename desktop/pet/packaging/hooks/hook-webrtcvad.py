from PyInstaller.utils.hooks import copy_metadata

# The maintained wheel exposes the import name ``webrtcvad`` while its
# distribution metadata is named ``webrtcvad-wheels``. PyInstaller's upstream
# hook still looks for the discontinued distribution name.
datas = copy_metadata("webrtcvad-wheels")
