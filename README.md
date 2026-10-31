 Step 1: Terminal mein ye commands run karo
Tumhara (.venv) already active hai.

Pehle libraries uninstall karo:
python -m pip uninstall numpy opencv-python mediapipe -y


Step 2: Compatible versions install karo


1. NumPy install:
python -m pip install numpy==1.26.4


2. OpenCV install:
python -m pip install opencv-python==4.10.0.84


3. MediaPipe install:
python -m pip install mediapipe==0.10.21


Step 3: Libraries check karo
python -c "import numpy, cv2, mediapipe as mp; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('MediaPipe:', mp.__version__); print('Solutions:', hasattr(mp, 'solutions'))"


Expected: Solutions: True aana chahiye.
Step 4: Project run karo
python main.py


Expected result