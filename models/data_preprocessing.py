import requests
import tarfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sagemaker
from sagemaker.session import Session
from sagemaker import get_execution_role
import os
import boto3




!mkdir ./train
!mkdir ./test


def extract_cifar_data(url, filename="cifar.tar.gz"):
    
    r = requests.get(url)
    with open(filename, "wb") as file_context:
        file_context.write(r.content)
    return

extract_cifar_data("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")    



with tarfile.open("cifar.tar.gz", "r:gz") as tar:
    tar.extractall()



with open("./cifar-100-python/meta", "rb") as f:
    dataset_meta = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/test", "rb") as f:
    dataset_test = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/train", "rb") as f:
    dataset_train = pickle.load(f, encoding='bytes')
    



label = set()
for n in range(len(dataset_train[b'data'])):
    if dataset_meta[b'fine_label_names'][dataset_train[b'fine_labels'][n]] == b'bicycle' or dataset_meta[b'fine_label_names'][dataset_train[b'fine_labels'][n]] == b'motorcycle':
        label.add(dataset_train[b'fine_labels'][n])

df_train = pd.DataFrame({
    "filenames": dataset_train[b'filenames'],
    "labels": dataset_train[b'fine_labels'],
    "row": range(len(dataset_train[b'filenames']))
})


df_train = df_train.drop(df_train[(df_train["labels"] != 8) & (df_train["labels"] != 48)].index, axis=0)


df_train["filenames"] = df_train["filenames"].apply(
    lambda x: x.decode("utf-8")
)


df_test = pd.DataFrame({
    "filenames": dataset_test[b'filenames'],
    "labels": dataset_test[b'fine_labels'],
    "row": range(len(dataset_test[b'filenames']))
})


df_test = df_test.drop(df_test[(df_test["labels"] != 8) & (df_test["labels"] != 48)].index, axis=0)

df_test["filenames"] = df_test["filenames"].apply(
    lambda x: x.decode("utf-8")
)


def save_images(df, index):
    
    if df is df_train:
        img = dataset_train[b'data'][index] 
    elif df is df_test:
        img = dataset_test[b'data'][index]
    
    
    target = np.dstack((
        img[0:1024].reshape(32,32),
        img[1024:2048].reshape(32,32),
        img[2048:].reshape(32,32)
    ))
    
    
    filename = df.loc[index, "filenames"]
    
    
    if df is df_train:
        plt.imsave(f"./train/{filename}", target)
    elif df is df_test: 
        plt.imsave(f"./test/{filename}", target)
    
    return


df_train.apply(lambda row: save_images(df_train, row.name), axis=1)
df_test.apply(lambda row: save_images(df_test, row.name), axis=1)



#Save data in AWS S3

role = get_execution_role()
session = sagemaker.Session()
region = session.boto_region_name
bucket = session.default_bucket()



os.environ["DEFAULT_S3_BUCKET"] = bucket
!aws s3 sync ./train s3://${DEFAULT_S3_BUCKET}/l2project/train/
!aws s3 sync ./test s3://${DEFAULT_S3_BUCKET}/l2project/test/



def to_metadata_file(df, prefix):
    df["s3_path"] = df["filenames"]
    df["labels"] = df["labels"].apply(lambda x: 0 if x==8 else 1)
    return df[["row", "labels", "s3_path"]].to_csv(
        f"{prefix}.lst", sep="\t", index=False, header=False
    )
    
to_metadata_file(df_train.copy(), "train")
to_metadata_file(df_test.copy(), "test")


boto3.Session().resource('s3').Bucket(
    bucket).Object('l2project/train.lst').upload_file('./train.lst')
boto3.Session().resource('s3').Bucket(
    bucket).Object('l2project/test.lst').upload_file('./test.lst')