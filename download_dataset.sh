#!/bin/bash
# Download TUM FR2 Pioneer SLAM3 dataset
# Run this script once after cloning the repo

DATASET_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz"
DATASET_DIR="rgbd_dataset_freiburg2_pioneer_slam3"

if [ -d "$DATASET_DIR" ]; then
    echo "Dataset already exists at $DATASET_DIR"
    exit 0
fi

echo "Downloading TUM FR2 Pioneer SLAM3 dataset (~1.4GB)..."
curl -L -O "$DATASET_URL"

echo "Extracting..."
tar xzf rgbd_dataset_freiburg2_pioneer_slam3.tgz

echo "Cleaning up archive..."
rm -f rgbd_dataset_freiburg2_pioneer_slam3.tgz

# Create symlink expected by some scripts
if [ ! -e "Dataset_VO" ]; then
    ln -s "$DATASET_DIR" Dataset_VO
fi

echo "Done! Dataset ready at $DATASET_DIR"
