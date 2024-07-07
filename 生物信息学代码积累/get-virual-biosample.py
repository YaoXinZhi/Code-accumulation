# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 26/05/2024 21:51
@Author: yao
"""
import os
from Bio import Entrez
import time

def read_id_file(id_file: str):

    id_set = set()
    with open(id_file) as f:
        for line in f:
            l = line.strip()
            id_set.add(l)
    print(f'{len(id_set):,} in {id_file}')
    return id_set


def fetch_sample_info(seq_id):
    handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="gb", retmode="text")
    record = handle.read()
    handle.close()
    return record

def main():
    id_file_path = '/mnt/disk1/xzyao/肺组织测序宏基因组/病毒序列下载/id_dir'
    save_path = '/mnt/disk1/xzyao/肺组织测序宏基因组/病毒序列下载/biosample_info_dir'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    id_file_list = [f'{id_file_path}/{file}' for file in os.listdir(id_file_path)]

    # 设置你的邮箱地址（NCBI要求用来跟踪查询）
    Entrez.email = "xinzhi_bioinfo@163.com"

    fail_count = 0
    for idx, id_file in enumerate(id_file_list):
        print(f'processing {idx+1}/{len(id_file_list)}: {id_file}')

        id_set = read_id_file(id_file)
        prefix = os.path.basename(id_file).split('.')[0]

        for seq_idx, seq_id in enumerate(id_set):
            print(f"Fetching data for {seq_idx}/{len(id_set)}: {prefix}-{seq_id}...")
            try:
                save_file = f'{save_path}/{seq_id}.{prefix}.txt'

                if os.path.exists(save_file):
                    continue

                sample_info = fetch_sample_info(seq_id)

                with open(save_file, 'w') as wf:
                    wf.write(f'> {seq_id}\n{sample_info}')
            except:
                print(f'fail-seq: {prefix}-{seq_id}')
                fail_count += 1
                # continue
        print(f'{prefix} seq saved, {fail_count}/{len(id_set)} failed.')

if __name__ == '__main__':


    main()
