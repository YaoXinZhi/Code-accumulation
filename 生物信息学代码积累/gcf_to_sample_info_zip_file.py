# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 28/06/2024 17:38
@Author: yao
"""

"""
从gcf到下载ncbi完整的样本信息
里面信息非常全 就是要再处理zip文件
包括WGS全序列鸟枪测序项目信息等
但是没有LOCUS
"""


import requests
import os
from tqdm import tqdm
import zipfile


def read_gcf_file(gcf_file: str):
    gcf_set = set()
    with open(gcf_file) as f:
        f.readline()
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            gcf = l[0]
            gcf_set.add(gcf)

    if '' in gcf_set:
        gcf_set.remove('')
    print(f'{len(gcf_set):,} GCFs.')
    return  gcf_set

def download_file(url, filename):
    # Stream the download in chunks for large files
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return ''
    except:
        # print(f'fail: {url}')
        return url



def extract_all_zips(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            zip_path = os.path.join(directory, filename)
            # 以压缩包名字命名文件夹
            folder_name = filename.replace(".zip", "")
            extract_path = os.path.join(directory, folder_name)

            # 创建目标文件夹
            os.makedirs(extract_path, exist_ok=True)

            # 解压缩文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            print(f"Extracted {filename} to {extract_path}")

def find_wgs(unzip_path: str):
    pass


def main():

    base_path = '/mnt/disk1/xzyao/Bacteria/小胖临时数据_240618/locus_download_628'
    # base_path = '/Users/yao/Downloads/小胖临时数据_240618/locus_download_628'

    gcf_file = f'{base_path}/Supplementary Data 2.txt'

    zip_save_path = f'{base_path}/2809-ncbi-data_dir'

    fail_file = f'{base_path}/fail-gcf.tsv'

    if not os.path.exists(zip_save_path):
        os.mkdir(zip_save_path)

    gcf_set = read_gcf_file(gcf_file)

    # print(gcf_set)
    # exit()
    fail_set = set()
    for gcfid in tqdm(gcf_set):
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v1/genome/accession/{gcfid}/download"

        save_file = f'{zip_save_path}/{gcfid}.zip'
        fail_url = download_file(url, save_file)
        if fail_url:
            fail_set.add(fail_url)

    if fail_set:
        with open(fail_file, 'w') as wf:
            fail_wf = '\n'.join(fail_set)
            wf.write(fail_wf)

    extract_all_zips(zip_save_path)



if __name__ == '__main__':
    main()
