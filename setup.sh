sudo apt install emacs
mkdir -p /root/.emacs.d/
echo '(global-set-key "\C-h" `delete-backward-char)' > /root/.emacs.d/init.el
echo '(global-set-key "\M-g" `goto-line)' >> /root/.emacs.d/init.el
git config --global user.name RyotaUshio
git config --global user.email gt0410ushi@g.ecc.u-tokyo.ac.jp
cd /content/drive/MyDrive/yolov5/
# git clone https://github.com/RyotaUshio/chestnut-detection
cd chestnut-detection
# git pull
chmod u+x make_dataset.py
chmod u+x data_split.py
chmod u+x make_yaml.py
cd ..
# git clone https://github.com/ultralytics/yolov5
cd yolov5
git pull
pip install -qr requirements.txt
cd ..
pip install --upgrade albumentations