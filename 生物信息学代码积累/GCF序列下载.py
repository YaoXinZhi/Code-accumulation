# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 01/06/2022 9:21
@Author: XINZHI YAO
"""

import os

def read_seq_id_file(seq_id_file: str):

    id_set = set()
    with open(seq_id_file) as f:
        f.readline()
        for line in f:
            id_set.add(line.strip())
    print(f'id count: {len(id_set):,}')
    return id_set

def main():

    seq_id_file = '2282.GCF.tsv'

    commend_file = f'ncbi-seq-download.sh'

    id_set = read_seq_id_file(seq_id_file)

    with open(commend_file, 'w') as wf:
        for gcf_id in id_set:
            commend = f'ncbi-genome-download --assembly-accessions {gcf_id} bacteria --section refseq --formats fasta --flat-output'
            wf.write(f'{commend}\n')
    print(f'{commend_file} saved.')


if __name__ == '__main__':

    main()
