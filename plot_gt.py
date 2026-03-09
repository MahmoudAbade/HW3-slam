import matplotlib.pyplot as plt
import numpy as np

def main():
    try:
        data = np.loadtxt('rgbd_dataset_freiburg2_pioneer_slam3/groundtruth.txt', comments='#')
        tx = data[:, 1]
        ty = data[:, 2]
        tz = data[:, 3]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(tx, ty, tz, label='Ground Truth Path')
        ax.set_xlabel('tx')
        ax.set_ylabel('ty')
        ax.set_zlabel('tz')
        plt.title('3D Ground Truth Trajectory')
        plt.legend()
        plt.savefig('ground_truth_path_3d.png')

        plt.figure(figsize=(10, 8))
        plt.plot(tx, ty, label='Ground Truth Path (tx vs ty)')
        plt.xlabel('tx')
        plt.ylabel('ty')
        plt.title('2D Ground Truth Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig('ground_truth_path_2d.png')
        print("Success! Plots saved.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
