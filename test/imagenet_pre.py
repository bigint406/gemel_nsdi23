# imagenet数据集预处理
import os
import tarfile
import scipy.io
import shutil

def move_val_img(val_dir='/mnt/data/zs/samba/datasets/imagenet-1k/val', devkit_dir='/mnt/data/zs/samba/datasets/imagenet-1k/ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
        /n01440764
            images
        /n01443537
            images
        .....
    """
    # load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

def untar_train_img():
    folder_path = "/mnt/data/zs/samba/datasets/imagenet-1k/train"

    # 遍历文件夹中的所有.tar文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tar"):
            # 创建相同名称的文件夹
            folder_name = os.path.splitext(file_name)[0]
            folder_path_new = os.path.join(folder_path, folder_name)
            os.makedirs(folder_path_new, exist_ok=True)

            # 解压.tar文件到新文件夹
            tar_path = os.path.join(folder_path, file_name)
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=folder_path_new)

            # 删除原始.tar文件
            os.remove(tar_path)
            # break

if __name__ == '__main__':
    # move_val_img()
    untar_train_img()