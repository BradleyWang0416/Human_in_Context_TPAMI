import os
import shutil

def delete_best_epochs_and_bin_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Delete folders named 'best_epochs'
        for dirname in dirnames:
            if dirname == 'best_epochs':
                folder_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print(f"Error deleting folder {folder_path}: {e}")

        # Delete files ending with '.bin'
        for filename in filenames:
            if filename.endswith('.bin'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

# Example usage
root_directory = '/data2/wxs/Skeleton-in-Context-tpami/ckpt2/'  # Replace with the path to your directory
delete_best_epochs_and_bin_files(root_directory)