# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 01/06/2022 9:21
@Author: XINZHI YAO
"""

import re
import os
import requests
import argparse

import time

from tqdm import tqdm

"""

"""

def get_bio_sample_set(summary_file: str):

    bio_sample_set = set()
    with open(summary_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')

            gcf_id = '_'.join(l[ -1 ].split('/')[ -1 ].split('_')[ :2 ])

            bio_sample = l[3]
            bio_sample_set.add((gcf_id, bio_sample))

    print(f'BioSample: {len(bio_sample_set)}')

    return bio_sample_set

def html_request(url: str):

    try:
        doc = requests.get(url)
    except:
        return ''

    if doc.status_code == 200:
        doc = doc.content
        doc = doc.decode('utf-8')
        return doc
    else:
        return ''

def save_doc(doc: str, save_file: str):
    with open(save_file, 'w') as wf:
        wf.write(doc)

def read_info(doc: str):

    strain = re.findall(r'<tr><th>strain</th><td>(.*?)</td></tr>', doc)
    # print(strain)
    strain = strain[0].strip() if strain else 'missing'

    collection_date = re.findall(r'<tr><th>collection date</th><td>(.*?)</td></tr>', doc)
    # print(collection_date)
    collection_date = collection_date[ 0 ].strip() if collection_date else 'missing'

    geographic_location = re.findall(r'<tr><th>geographic location</th><td>(.*?)</td></tr>', doc)
    # print(geographic_location)
    if geographic_location:
        geographic_location = geographic_location[0].strip()
        geographic_location = re.findall(r'>(.*)</a>', geographic_location)
    # print(geographic_location)
    geographic_location = geographic_location[ 0 ].strip() if geographic_location else 'missing'

    host = re.findall(r'<tr><th>host</th><td>(.*?)</td></tr>', doc)
    # print(host)
    host = host[ 0 ].strip() if host else 'missing'

    host_dis = re.findall(r'<tr><th>host disease</th><td>(.*?)</td></tr>', doc)
    # print(host_dis)
    host_dis = host_dis[ 0 ].strip() if host_dis else 'missing'

    isolation_source = re.findall(r'<tr><th>isolation source</th><td>(.*?)</td></tr>', doc)
    # print(isolation_source)
    isolation_source = isolation_source[ 0 ].strip() if isolation_source else 'missing'

    latitude_and_longitude = re.findall(r'<tr><th>latitude and longitude</th><td>(.*?)</td></tr>', doc)
    # print(latitude_and_longitude)
    latitude_and_longitude = latitude_and_longitude[ 0 ].strip() if latitude_and_longitude else 'missing'

    sub_species = re.findall(r'<tr><th>sub species</th><td>(.*?)</td></tr>', doc)
    # print(sub_species)
    sub_species = sub_species[ 0 ].strip() if sub_species else 'missing'

    serotype = re.findall(r'<tr><th>serotype</th><td>(.*?)</td></tr>', doc)
    # print(serotype)
    serotype = serotype[ 0 ].strip() if serotype else 'missing'

    encoded_traits = re.findall(r'<tr><th>encoded traits</th><td>(.*?)</td></tr>', doc)
    # print(encoded_traits)
    encoded_traits = encoded_traits[ 0 ].strip() if encoded_traits else 'missing'

    environmental_medium = re.findall(r'<tr><th>environmental medium</th><td>(.*?)</td></tr>', doc)
    # print(environmental_medium)
    environmental_medium = environmental_medium[ 0 ].strip() if environmental_medium else 'missing'

    genotype = re.findall(r'<tr><th>genotype</th><td>(.*?)</td></tr>', doc)
    # print(genotype)
    genotype = genotype[ 0 ].strip() if genotype else 'missing'

    return strain, collection_date, geographic_location, host, \
           host_dis, isolation_source, latitude_and_longitude, \
            sub_species, serotype, encoded_traits, environmental_medium, \
            genotype


def read_complete_gcf(complete_file: str):

    complete_set = set()
    with open(complete_file) as f:
        for line in f:
            l = line.strip().split('\t')
            gcf_id = l[0]
            complete_set.add(gcf_id)
    return complete_set



def main(summary_file: str, save_path: str, complete_file: str):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    html_save_path = f'{save_path}/html_dir'

    if not os.path.exists(html_save_path):
        os.mkdir(html_save_path)

    if complete_file:
        complete_set = read_complete_gcf(complete_file)
    else:
        complete_set = set()

    save_file = f'{save_path}/BioSample.download.tsv'
    miss_file = f'{save_path}/miss.tsv'

    bio_sample_set = get_bio_sample_set(summary_file)

    bio_sample_set = bio_sample_set - complete_set

    count = 0
    with open(save_file, 'w') as wf, open(miss_file, 'w') as wf_miss:
        wf.write(f'GCF id\tstrain\tcollection_date\tgeographic_location\thost\thost_dis\tisolation_source\tlatitude_and_longitude\tsub_species\tserotype\tencoded_traits\tenvironmental_medium\tgenotype\n')
        wf_miss.write(f'GCF id\tBioSample\n')
        for gcf_id, bio_sample in tqdm(bio_sample_set):
            
            # if gcf_id in complete_set:
            #     continue

            count += 1
            if count % 100 == 0:
                time.sleep(18)

            url = f'https://www.ncbi.nlm.nih.gov/biosample/{bio_sample}/'
            # print(gcf_id)
            # print(bio_sample)
            # print(url)
            # input()
            doc = html_request(url)

            if not doc:
                wf_miss.write(f'{gcf_id}\t{bio_sample}\n')
                continue

            html_save_file = f'{html_save_path}/{bio_sample}.txt'
            save_doc(doc, html_save_file)

            strain, collection_date, geographic_location, host, \
            host_dis, isolation_source, latitude_and_longitude, \
            sub_species, serotype, encoded_traits, environmental_medium, \
            genotype = read_info(doc)

            # print(strain, collection_date, geographic_location, host,
            # host_dis, isolation_source, latitude_and_longitude,
            # sub_species, serotype, encoded_traits, environmental_medium,
            # genotype)
            # input()

            wf.write(f'{gcf_id}\t{strain}\t{collection_date}\t'
                     f'{geographic_location}\t{host}\t{host_dis}\t'
                     f'{isolation_source}\t{latitude_and_longitude}\t'
                     f'{sub_species}\t{serotype}\t{encoded_traits}\t'
                     f'{environmental_medium}\t{genotype}\n')

    print(f'{save_file} save done.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bulk_download_for_BioSample')
    parser.add_argument('-if', dest='summary_file',
                        default='../data/prokaryotes.tsv',
                        help="default: ../data/prokaryotes.tsv")
    parser.add_argument('-sf', dest='save_path',
                        required=True)

    parser.add_argument('-cf', dest='complete_file',
                        default='', required=False)
    args = parser.parse_args()

    # summary_file = '../data/prokaryotes.tsv'
    #
    # save_path = '../data/BioSample_dir'

    main(args.summary_file, args.save_path, args.complete_file)

