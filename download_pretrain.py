import os
import urllib.request
import tarfile


def main(files):
    if not os.path.isdir("./pretrained-models/"):
        os.mkdir("./pretrained-models/")

    urllib.request.urlretrieve(
        "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
        + files
        + ".tar.gz",
        "./pretrained-models/" + files + ".tar.gz",
    )

    # open file
    f = tarfile.open("./pretrained-models/" + files + ".tar.gz")

    # extracting file
    f.extractall("./pretrained-models/")
    f.close()


if __name__ == "__main__":
    files = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
    main(files)
