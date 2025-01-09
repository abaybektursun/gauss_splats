sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libsdl2-dev

sudo apt-get update && sudo apt-get install -y \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx

sudo apt-get install libglm-dev


# Nsight
sudo apt-get update
sudo apt-get install -y \
    libnss3 \
    libnspr4 \
    libnss3-tools \
    libxcomposite1 \
    libxdamage1 \
    libxtst6 \
    libxkbfile1 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-dri2-0 \
    libxcb-dri3-0 \
    libxcb-present0 \
    libxcb-sync1 \
    libxshmfence1 \
    libglx0 \
    libgl1 \
    libglvnd0 \
    libxcursor1 \
    libxrandr2 \
    libxi6 \
    libcups2 \
    libfontconfig1 \
    libfreetype6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Train
sudo apt-get install libopencv-dev -y