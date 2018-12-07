#coding=utf-8

#使用这个py文件读xgb的dump文件进行解析，生成固定格式，后续接xgb_dump_recover.java文件

import os, sys, datetime, re

booster_id = -1
ps = ''
useful_feature = []
##修改
module_fp = open("xgb_dump_result.txt","r")

##修改end
##for line in sys.stdin:
with open('xgb_dump_recover.data','w') as f:
    for line in module_fp:
        line = line.strip()
        if line.startswith('booster'):
            res = re.findall(r'booster\[([^\]]+)', line)
            if len(res) == 0:
                continue
            booster_id = res[0]
            if int(booster_id)>49:
                break
            ps += 'booster;' + booster_id + ';'
            continue
        arr = line.split(':')
        node_index = arr[0].strip()
        res = re.findall(r'\[([^<]+)<([^\]]+).+yes=([0-9]+).+no=([0-9]+).+missing=([0-9]+)', line)
        if len(res) != 0:
            fea_idx = res[0][0][0:]
            fea_value = res[0][1]
            yes_id = res[0][2]
            no_id = res[0][3]
            missing_id = res[0][4].strip()
            useful_feature.append(fea_idx)
            ps += str(node_index) + ';node;' + fea_idx + ':' + str(
                fea_value) + ';' + yes_id + ';' + no_id + ';' + missing_id + ';'
            f.write(str(
                booster_id) + '\tnode\t' + node_index + '\t' + fea_idx + '\t' + fea_value + '\t' + yes_id + '\t' + no_id + '\t' + missing_id+'\n')
            continue
        res = re.findall(r'leaf=(.+)', line)
        if len(res) != 0:
            leaf_value = res[0].strip()
            ps += node_index + ';leaf;' + leaf_value + ';'
            f.write(str(booster_id) + '\tleaf\t' + node_index + '\t' + leaf_value+'\n')

'''
fin = open('./xgboost_feature_extract/data/feature_map.dat')
fea_map = {}
for line in fin:
    arr = line.split('\t')
    fea_id = arr[0].strip()
    fea_name = arr[1].strip()
    fea_map[fea_id] = fea_name
fin.close()


fea_map = {}

useful_feature = list(set(useful_feature))

for fea in useful_feature:
    fea_map[fea] = fea

fea_idx_s = ''
for idx in useful_feature:
    fea_name = fea_map[idx]
    fea_idx_s += idx + ':' + fea_name + '|'
fea_idx_s = fea_idx_s.strip('|')

ps = ps.strip(';')
fout = open('./model/m_ftrl_xgboost_model.dat', 'w')
fout.write('xgboost_model\t' + ps + '\n')
fout.write('feature_idx\t' + fea_idx_s + '\n')
fout.close()

'''
