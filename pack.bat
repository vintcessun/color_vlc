set MPLBACKEND=Agg

nuitka --standalone --onefile --jobs=6 ^
 --plugin-enable=torch ^
 --plugin-enable=numpy ^
 --plugin-enable=matplotlib ^
 --include-package=qreader ^
 --include-package-data=qreader ^
 --show-progress ^
 --assume-yes-for-downloads ^
 --nofollow-import-to=tkinter ^
 --nofollow-import-to=yt_dlp ^
 --nofollow-import-to=ipywidgets ^
 --nofollow-import-to=selenium ^
 --nofollow-import-to=tornado ^
 --nofollow-import-to=bokeh ^
 --nofollow-import-to=tensorboard ^
 --nofollow-import-to=numba ^
 --nofollow-import-to=llvmlite ^
 --nofollow-import-to=sqlalchemy ^
 --nofollow-import-to=tensorflow ^
 --nofollow-import-to=keras ^
 --nofollow-import-to=pandas ^
 --nofollow-import-to=statsmodels ^
 --nofollow-import-to=jupyter_lab ^
 --nofollow-import-to=jupyter_server ^
 --nofollow-import-to=notebook ^
 --nofollow-import-to=nbformat ^
 --nofollow-import-to=google ^
 --nofollow-import-to=git ^
 --nofollow-import-to=cryptography ^
 --nofollow-import-to=tornado ^
 --nofollow-import-to=zmq ^
 --include-package-data=ultralytics ^
 --include-package-data=qrdet ^
 decoder.py