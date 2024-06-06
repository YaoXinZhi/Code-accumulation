# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 06/06/2024 09:32
@Author: yao
"""

"""
该代码用于根据mesh的heading获取mesh的unique ID
实际上不是准确的ID，但是是一类MeSH id
https://meshb.nlm.nih.gov/record/ui?ui=D008659
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term=Metabolic%20Diseases
"""

import os

import requests
from bs4 import BeautifulSoup


def get_mesh_unique_id(mesh_heading):
    # 使用esearch查找MeSH term的UID
    esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={mesh_heading.replace(' ', '%20')}&retmode=json"
    esearch_response = requests.get(esearch_url)
    esearch_data = esearch_response.json()

    if 'esearchresult' in esearch_data and 'idlist' in esearch_data[ 'esearchresult' ] and \
            esearch_data[ 'esearchresult' ][ 'idlist' ]:
        mesh_uid = esearch_data[ 'esearchresult' ][ 'idlist' ][ 0 ]

        # 使用esummary获取MeSH term的详细信息
        esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id={mesh_uid}&retmode=json"
        esummary_response = requests.get(esummary_url)
        esummary_data = esummary_response.json()

        if 'result' in esummary_data and mesh_uid in esummary_data[ 'result' ]:
            mesh_details = esummary_data[ 'result' ][ mesh_uid ]
            unique_id = mesh_details.get('uid', None)
            return unique_id
    return '-'

def read_detail_file(detail_file: str):

    id_to_name = {}
    name_to_id = {}
    with open(detail_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            mesh_id = l[0]
            mesh_name = l[1]

            id_to_name[mesh_id] = mesh_name
            name_to_id[mesh_name] = mesh_id
    return id_to_name, name_to_id


def read_ancestor_file(ancestor_file: str):

    ancestor_name_set = set()
    with open(ancestor_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            if len(l) < 3:
                continue
            mesh_name = l[2]
            ancestor_name_set.add(mesh_name)
    print(f'{len(ancestor_name_set):,} ancestor names loaded.')
    return ancestor_name_set

def main():
    base_path = '/mnt/disk1/xzyao/AD-PNRLE/AD-Alterome调整-2024/result/top_element_result'

    ancestor_file = f'{base_path}/mesh-tree.third.tsv'
    # ancestor_file = f'{base_path}/mesh-tree.tsv'

    detail_file = f'{base_path}/mesh-alterome.detail.tsv'

    save_file = f'{base_path}/mesh-ancestor_id.tsv'

    id_to_name, name_to_id = read_detail_file(detail_file)

    ancestor_name_set = read_ancestor_file(ancestor_file)
    with open(save_file, 'w') as wf:
        wf.write(f'AncestorID\tAncestorName\n')
        for idx, ancestor_name in enumerate(ancestor_name_set):

            if idx % 20 == 0 and idx != 0:
                print(f'{idx}/{len(ancestor_name_set):,} ancestor processed.')

            if name_to_id.get(ancestor_name):
                ancestor_id = name_to_id[ancestor_name]
            else:
                ancestor_id = get_mesh_unique_id(ancestor_name)
                if ancestor_id != '-':
                    ancestor_id = f'D{ancestor_id[-6:]}'
            wf.write(f'{ancestor_id}\t{ancestor_name}\n')
            wf.flush()

    print(f'{save_file} saved.')


if __name__ == '__main__':
    main()
