import os
import shutil
import urllib.request
import zipfile

def download_tiny_imagenet():

    if os.path.exists(os.path.join("tiny-imagenet", "tiny-imagenet-200")):
        print("Dataset già presente, nessuna azione necessaria.")
        return

    # ======== DOWNLOAD E ESTRAZIONE =========
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"


    if not os.path.exists(zip_path):
        print("Scarico il dataset...")
        urllib.request.urlretrieve(url, zip_path)

    print("Estrazione del file zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tiny-imagenet")

    # ======== SISTEMAZIONE DELLA CARTELLA VAL =========
    val_dir = 'tiny-imagenet/tiny-imagenet-200/val'

    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            class_dir = os.path.join(val_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copyfile(
                os.path.join(val_dir, 'images', fn),
                os.path.join(class_dir, fn)
            )

    shutil.rmtree(os.path.join(val_dir, 'images'))

    print("Dataset TinyImageNet pronto!")


if __name__ == "__main__":
    download_tiny_imagenet()