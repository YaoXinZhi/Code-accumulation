# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 05/06/2024 16:36
@Author: yao
"""


"""
该代码用于爬取MeSH数据库
https://meshb-prev.nlm.nih.gov/record/ui?ui=D028361
其中的
MeSH Heading Mitochondrial Diseases
Tree Number(s) C18.452.660
Unique ID D028361
RDF Unique Identifier http://id.nlm.nih.gov/mesh/D028361

Nutritional and Metabolic Diseases [C18]
Metabolic Diseases [C18.452]
Acid-Base Imbalance [C18.452.076] 
Bone Diseases, Metabolic [C18.452.104] 
Brain Diseases, Metabolic [C18.452.132] 
Calcium Metabolism Disorders [C18.452.174] 
DNA Repair-Deficiency Disorders [C18.452.284] 
Glucose Metabolism Disorders [C18.452.394] 
Hyperlactatemia [C18.452.479]
Iron Metabolism Disorders [C18.452.565] 
Lipid Metabolism Disorders [C18.452.584] 
Malabsorption Syndromes [C18.452.603] 
Metabolic Syndrome [C18.452.625]
Metabolism, Inborn Errors [C18.452.648] 
Mitochondrial Diseases [C18.452.660] 
"""
import time
import re
import requests
from icecream import ic
from bs4 import BeautifulSoup

def read_mesh_id(mesh_file: str):

    mesh_id_set = set()
    with open(mesh_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            mesh_id = l[0]

            if mesh_id.startswith('MESH'):
                mesh_id = mesh_id.split(':')[1]
            else:
                raise ValueError(mesh_id)

            mesh_id_set.add(mesh_id)
    print(f'total {len(mesh_id_set):,} mesh ids.')
    return mesh_id_set

def re_info_finder(document: str):

    matches = re.findall(r'<dt>(.*?)</dt>\s*<dd>(.*?)</dd>', document, re.DOTALL)

    heading = ''
    tree_num = ''
    unique_id = ''
    rdf_id = ''
    supplementary_concept = 'False'
    for match in matches:
        dt = match[0].strip()
        dd = match[1].strip()
        if dt == "MeSH Heading":
            heading = dd
        elif dt == "Tree Number(s)":
            tree_matches = re.findall(r'>(.*?)</a>', dd)
            tree_num = tree_matches[0]
        elif dt == "Unique ID":
            unique_id = dd
        elif dt == "RDF Unique Identifier":
            rdf_id = dd
        elif dt == 'MeSH Supplementary' and not heading:
            heading = dd
            supplementary_concept = True

    # print(f"MeSH Heading: {heading}")
    # print(f"Tree Number(s): {tree_num}")
    # print(f"Unique ID: {unique_id}")
    # print(f"RDF Unique Identifier: {rdf_id}")
    # print(f'supplementary_concept: {supplementary_concept}')
    return heading, tree_num, unique_id, rdf_id, supplementary_concept

def re_find_ancestor_info(ancestor_id: str, doc: str):
    pattern = r'<span>(.*?)\[{}\]</span>'.format(ancestor_id)
    matches = re.findall(pattern, doc)

    if matches:
        ancestor_name = matches[0]
    else:
        ancestor_name = ''
    return ancestor_name

def mesh_info_bulk_download(mesh_id_set: set, mesh_id_save_file: str,
                            tree_save_file: str, fail_mesh_id_save_file: str):

    fail_mesh_id_set = set()
    wf_id = open(mesh_id_save_file, 'w')
    wf_id.write('ID\theadin\tTreeNum\tRDF-ID\tSupplementaryConcept\n')
    wf_tree = open(tree_save_file, 'w')
    wf_tree.write(f'MeshID\tAncestorID\tMeSHName\n')
    for idx, mesh_id in enumerate(mesh_id_set):
        if idx % 20 == 0 and idx != 0:
            print(f'{idx:,}/{len(mesh_id_set):,} processed.')
            print(f'{len(fail_mesh_id_set):,}/{idx:,} failed.')

        url = f'https://meshb-prev.nlm.nih.gov/record/ui?ui={mesh_id}'
        response = requests.get(url)
        # print(url)
        if response.status_code == 200:
            document = response.text
            heading, tree_num, unique_id, rdf_id, supplementary_concept = re_info_finder(document)
            wf_id.write(f'{unique_id}\t{heading}\t{tree_num}\t{rdf_id}\t{supplementary_concept}')

            ancestor_tree_id = '.'.join(tree_num.split('.')[:2])
            # print(f'ancestor_tree_id: {ancestor_tree_id}')

            if ancestor_tree_id == unique_id:
                ancestor_name = heading
            else:
                ancestor_name = re_find_ancestor_info(ancestor_tree_id, document)
                # print(f'ancestor_name: {ancestor_name}')
            wf_tree.write(f'{unique_id}\t{ancestor_tree_id}\t{ancestor_name}\n')
        else:
            fail_mesh_id_set.add(mesh_id)


    with open(fail_mesh_id_save_file, 'w') as wf:
        for idx in fail_mesh_id_set:
            wf.write(f'{idx}\n')

    print(f'{len(fail_mesh_id_set)}/{len(mesh_id_set):,} ids failed.')


def main():
    mesh_path = '/mnt/disk1/xzyao/AD-PNRLE/AD-Alterome调整-2024/result/top_element_result'

    mesh_file = f'{mesh_path}/mesh-sorted.tsv'

    mesh_id_save_file = f'{mesh_path}/mesh-alterome.detail.tsv'
    tree_save_file = f'{mesh_path}/mesh-tree.tsv'
    fail_mesh_id_save_file = f'{mesh_path}/fail-mesh.tsv'

    mesh_id_set = read_mesh_id(mesh_file)

    mesh_info_bulk_download(mesh_id_set, mesh_id_save_file, tree_save_file, fail_mesh_id_save_file)

if __name__ == '__main__':
    main()
