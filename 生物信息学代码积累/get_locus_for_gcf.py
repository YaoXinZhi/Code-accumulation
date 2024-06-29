# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 29/06/2024 08:47
@Author: yao
"""

"""
这个代码是真的能直接从GCF获取LOCUS
"""

from Bio import Entrez
from Bio import SeqIO
import requests
import os
from tqdm import tqdm
import zipfile

Entrez.email = "xinzhi_bioinfo@163.com"

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


def get_locus_from_gcf(gcf_id):

    # 使用Entrez esearch工具查找基因组ID
    handle = Entrez.esearch(db="nuccore", term=gcf_id)
    record = Entrez.read(handle)
    handle.close()

    if not record[ 'IdList' ]:
        return None

    nuc_id = record[ 'IdList' ][ 0 ]

    # 使用Entrez efetch工具获取基因组信息
    handle = Entrez.efetch(db="nuccore", id=nuc_id, rettype="gb", retmode="text")
    records = list(SeqIO.parse(handle, "genbank"))
    handle.close()

    if records:
        return records[ 0 ].name  # LOCUS名通常在GenBank记录的name属性中

    return None

def main():

    base_path = '/mnt/disk1/xzyao/Bacteria/小胖临时数据_240618/locus_download_628'
    # base_path = '/Users/yao/Downloads/小胖临时数据_240618/locus_download_628'
    gcf_file = f'{base_path}/Supplementary Data 2.txt'
    save_file = f'{base_path}/2809-locus.tsv'

    gcf_set = read_gcf_file(gcf_file)

    with open(save_file, 'w') as wf:
        wf.write('GCF ID\tLOCUS\n')
        for gcfid in tqdm(gcf_set):

            locus = get_locus_from_gcf(gcfid)

            if locus:
                wf.write(f'{gcfid}\t{locus}\n')
            else:
                wf.write(f'{gcfid}\t-\n')
            wf.flush()
    print(f'{save_file} saved.')


if __name__ == '__main__':
    main()


