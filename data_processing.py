import pandas as pd
import numpy as np

###############################
#read csv
table = pd.read_csv("xgb_train.csv")
#show the head lines
table_all.head()
#change one column value
table['y']=table['click_time'].apply(lambda x: 0 if pd.isnull(x) else 1)  #用apply来遍历一列
###############################



###############################
#缺失值删除

table_new=pd.DataFrame()
for i in table.columns:
    if table[i].isnull().sum()<1500000:
        table_new[i]=table[i]
table_new.shape
###############################



###############################
#缺失值统计
count_nan = []
for i in table.columns:
    count = table[i].isnull().sum()
    count_nan.append([i,count/3000000.0])
count_nan = sorted(count_nan,key=lambda x:x[1],reverse = True)
###############################



###############################
#方差选择 , 选择小于0.01的
std = table_new.std()
std.index
std_less=[]
feature_list = std.index

for i in range(len(std)):
    if std[i]<0.01:
        std_less.append(feature_list[i])
print(std_less)
###############################

###############################
#等距分箱
score_list=['a','b','c']

X_hot[score_list].fillna(0)
bins = [0,0.2,0.4,0.6,0.8,1.0]
for i in score_list:   
    X_hot[i+'_box'] = pandas.cut(X_hot[i], bins,labels=['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    X_hot=pandas.concat([X_hot,pandas.get_dummies(X_hot[i+'_box'],dummy_na=True).rename(columns=lambda x:i+'_box'+':'+str(x).replace(" ", "_"))],axis=1)
    X_hot = X_hot.drop([i+'_box'],axis=1)
X_hot=X_hot.drop(score_list,axis=1)

################################



###############################
#卡方分箱
import pandas as pd
import numpy as np

'''
example
data = pd.read_csv('E:/breast_cancer.csv', sep=',')
temp = data[['radius_mean','diagnosis']]
temp2=ChiMerge(temp,'radius_mean' , 'diagnosis',confidenceVal=5.841, bin=5, sample = None)
'''

# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
'''
    运行前需要 import pandas as pd 和 import numpy as np
    df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    variable:需要卡方分箱的变量名称（字符串）
    flag：正负样本标识的名称（字符串）
    confidenceVal：置信度水平（默认是不进行抽样95%）
    bin：最多箱的数目
    sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
'''	
def ChiMerge(df,result_data, variable, flag, confidenceVal=3.841, bin=10, sample = None):  
#进行是否抽样操作
    if sample != None:
        df = df.sample(n=sample)
    else:
        df 
		
#进行数据格式化录入
    total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                       how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    print(variable+': reading')

#处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1
        
 #对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
          * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
          ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
          np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)
    print(variable+': mergeing')

#把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                   ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    print(variable+': done')

#把结果保存成一个数据框
    #result_data = pd.DataFrame()  # 创建一个保存结果的数据框
    #result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列：变量名
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            #x = '0' + ',' + str(np_regroup[i, 0])
            x = 0
        elif i == np_regroup.shape[0] - 1:
            #x = str(np_regroup[i - 1, 0]) + '+'
            x = np_regroup[i - 1, 0]
        else:
            #x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
            x = np_regroup[i - 1, 0]
        list_temp.append(x)
    #result_data['interval'] = list_temp  # 结果表第二列：区间
    #result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
    #result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
    while len(list_temp)<bin:
        list_temp.insert(0,0.0);
    result_data[variable] = list_temp
    
    #return result_data


#卡方分箱使用
fea_2chi = ['a','b','c']
result_chi_total =pd.DataFrame()

for i in fea_2chi:
    temp = table_all[[i,'y']]
    ChiMerge(temp,result_data_1, i, 'y', bin=5, sample = None)

# result_chi_total中是a,b,c三列，需要继续手工onehot，可能前面有重复的0
table_tmp = pd.DataFrame()
for fea in result_chi_total.columns:
    box_list = []
    for i in result_chi_total[fea]:
        box_list.append(i)
    
    box_list.append(table_all[fea].max())
    
    box_set = sorted(list(set(box_list)))
    
    label_list = []
    for index in range(len(box_set)-1):
        if index==0:
            label_list.append('0_'+str(box_set[index+1]))
        else:
            label_list.append(str(box_set[index])+'_'+str(box_set[index+1]))
            
    #print label_list
    
    table_tmp[fea+'_box'] = pd.cut(table_all[fea], box_set , labels=label_list)
    table_tmp = pd.concat([table_tmp,pd.get_dummies(table_tmp[fea+'_box'],dummy_na=True).rename(columns=lambda x:fea+'_box'+':'+str(x).replace(" ", "_"))],axis=1)
    table_tmp = table_tmp.drop(fea+'_box',axis=1)
############################### 


 
############################### 
#将一些特征one-hot编码
#先创造一个新的dataFrame再往上拼接

hot_list=['a','b','c']
for hot in hot_list:
    table_tmp=pd.concat([table_tmp,pd.get_dummies(table_all[hot],dummy_na=True,dtype=int).rename(columns=lambda x:hot+':'+str(x).replace(" ", "_"))],axis=1)
table_tmp.head()
############################### 



#################################
#看相关,大于正负0.0099？
corrDf = table.corr()
corr_list = corrDf['y'].sort_values(ascending =False)
feature_list = corr_list.index
cor_to_drop=[]
for i in range(len(corr_list)):
    if abs(corr_list[i])<0.001:
        cor_to_drop.append(feature_list[i])
#打印出来
for j in cor_to_drop:
    print j.decode('utf-8')

outfile = open('corr.txt', 'w')
#或写出到文件
feature_corr = corr_list.index
for f in range(len(corr_list)):
    outfile.write("%2d) %-*s %f" % (f + 1, 70, feature_corr[f], corr_list[f]))
    outfile.write("\n")
outfile.close()
###############################
