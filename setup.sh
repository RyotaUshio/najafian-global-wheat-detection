cd /content/drive/MyDrive/
# git clone https://github.com/RyotaUshio/chestnut-detection
cd chestnut-detection
# git pull
# git clone https://github.com/ultralytics/yolov5
cd yolov5
# git pull
pip install -qr requirements.txt
cd ..
# pip install --upgrade albumentations
pip uninstall opencv-python-headless==4.5.5.62 -y
pip install opencv-python-headless==4.1.2.30
cd pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt
cd ..
