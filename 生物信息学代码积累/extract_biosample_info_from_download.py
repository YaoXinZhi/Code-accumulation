# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 27/05/2024 16:22
@Author: yao
"""

"""
该代码用于从 get-virual-biosample.py下载的文件中
吧source里面的样本信息都抽取出来
"""

import os
import re
from collections import defaultdict


def parse_string_to_dict(input_string):
    # 使用正则表达式匹配键值对
    pattern = r'/(\S+)="([^"]+)"'
    matches = re.findall(pattern, input_string)

    # 将匹配到的键值对转换为字典
    result_dict = {key: value for key, value in matches}
    return result_dict


def main():

    biosample_info_path = '/mnt/disk1/xzyao/肺组织测序宏基因组/病毒序列下载/biosample_info_dir'

    file_list = [f'{biosample_info_path}/{file}' for file in os.listdir(biosample_info_path)]

    save_path = '/mnt/disk1/xzyao/肺组织测序宏基因组/病毒序列下载/biosample_info_processed'
    save_file = f'{save_path}/viral-biosample-info.tsv'

    # sample_id: {'': '', }
    sample_to_info = defaultdict(dict)

    fail_sample_id_set = set()
    total_attr_set = set()
    for file in file_list:
        # print(file)
        # input()
        with open(file) as f:

            base_name = os.path.basename(file)
            sample_id = '.'.join(base_name.split('.')[:-2])
            sample_group = base_name.split('.')[-2].split('-')[-2]

            start_bool = False
            context = ''
            # 换行的
            quote_count = 0
            for line in f:
                l = line.strip().split()
                if len(l) < 1:
                    continue
                # 开始行
                if l[0] == 'source':
                    start_bool = True
                    continue
                # 结束行
                if start_bool and '=' not in line and quote_count%2==0:
                    break
                if start_bool:
                    context += line.strip()
                    context += ' '
                    quote_count += line.count('"')
        # print(context)
        # print(context.count('='))
        # input()
        result_dict = parse_string_to_dict(context)
        # print(result_dict)
        # print(len(result_dict))
        # input()
        if not result_dict:
            fail_sample_id_set.add(sample_id)
            continue

        sample_to_info[sample_id] = result_dict
        sample_to_info[sample_id]['sample_group'] = sample_group
        total_attr_set.update(sample_to_info[sample_id].keys())
        # print(sample_to_info)
        # print(total_attr_set)
        # input()
    save_biosample_info(sample_to_info, total_attr_set, save_file)
    print(f'fail sample: {fail_sample_id_set}.')


def save_biosample_info(sample_to_info: dict, total_attr_set: set,
                        save_file: set):

    sorted_attr = list(sorted(total_attr_set))
    sorted_sample = list(sorted(sample_to_info.keys()))

    with open(save_file, 'w') as wf:
        head = ['Sample ID']
        head.extend(sorted_attr)
        head_wf = '\t'.join(head)
        wf.write(f'{head_wf}\n')

        for sample_id in sorted_sample:
            sample_info_dict = sample_to_info[sample_id]
            line_wf = [sample_id]

            for attr in sorted_attr:
                if sample_info_dict.get(attr):
                    line_wf.append(sample_info_dict[attr])
                else:
                    line_wf.append('-')

            line_wf = '\t'.join(line_wf)
            wf.write(f'{line_wf}\n')
    print(f'{save_file} saved.')




if __name__ == '__main__':
    main()
