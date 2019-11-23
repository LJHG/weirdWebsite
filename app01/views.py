from django.shortcuts import render
from django.shortcuts import HttpResponse
import json
import numpy as np
import pandas as pd
import os
# pieces = []
# for year in range(1880,2011):
#     frame = pd.read_csv('../../babynames/yob%d.txt'%year,names=['name','gender','frequency']) #用这个比loadtxt再转dataframe快
#     frame['year']=year
#     #print(frame)
#     #print(frame['frequency'].sum())
#     pieces.append(frame)
# #print(pieces)   
# names = pd.concat(pieces,ignore_index=True)#ignore_index=true 重新索引

names = None
data = None
users_info = None
user_item = None


def read_data():
    global names
    pieces = []
    for year in range(1880,2011):
        frame = pd.read_csv('app01/static/babynames/yob%d.txt'%year,names=['name','gender','frequency']) #注意这里的路径是相对于manage.py的
        frame['year']=year
        #print(frame)
        #print(frame['frequency'].sum())
        pieces.append(frame) 
    names = pd.concat(pieces,ignore_index=True)#ignore_index=true 重新索引
    return

def get_births(name,year):
    #对应goto_b1
    tmpframe = names[(names['name']==name) & (names['year']==year)] #设定条件进行查询
    ans = tmpframe['frequency'].sum()  #对某一列求和 .sum后面要打()
    return(ans)

def get_births_years(name,start_year,end_year):
    #输入姓名、开始年份和结束年份，绘制该姓名在各年份出生人数折线图；
    frame = names[(names['name']==name) & (names['year']>=start_year) & (names['year']<=2010)]
    df=frame.groupby('year').sum()
    return df.index.values.tolist(),df.values.flatten().tolist()

def multi_person_births(name,start_year,end_year):
    namesum=[]
    for item in name:
        frame = names[(names['name']==item) & (names['year']>=start_year) & (names['year']<=2010)]
        df=frame.groupby('year').sum()
        namesum.append(df['frequency'].sum())
    return namesum

def survivor(name,start_year,end_year,lifespan):
    alive = []
    tmpframe = names[(names['name']==name) & (names['year']>=(start_year-lifespan)) & (names['year']<=end_year)]#先筛选需要用到的数据
    year=tmpframe.groupby('year').sum()

    for i in range(start_year,end_year+1):
        #print(i)
        if(i<1880):
            ans =0
        elif(i>=1880 and (i-lifespan) <1880):
            ans = year.loc[1880:i]['frequency'].sum()
        else:
            ans = year.loc[i-lifespan:i]['frequency'].sum()
        alive.append(ans)
    year = np.linspace(start_year,end_year,end_year-start_year+1)
    return year.tolist(),alive

def Correlation(namea,nameb,start_year,end_year):
    tmpa = names[(names['name']==namea) & (names['year']>=start_year) & (names['year']<=end_year)]
    a=tmpa.groupby('year').sum()
    tmpb = names[(names['name']==nameb) & (names['year']>=start_year) & (names['year']<=end_year)]
    b=tmpb.groupby('year').sum()
    correlation = a.corrwith(b).values[0]
    year = a.index.values.tolist()
    return a.values.flatten().tolist(),b.values.flatten().tolist(),correlation,year

def births_oneyear_sort(year):
    tmp = names[ (names['year']==year)]
    dict ={}
    for index, row in tmp.iterrows():
        if(row['name'] in dict):
            dict[row['name']]= dict[row['name']] + row['frequency']
        else:
            dict[row['name']]= row['frequency']
    ans = pd.DataFrame.from_dict(dict, orient='index', columns=['values'])
    ans = ans.sort_values(by = 'values',ascending = False)
    return ans.index.tolist(),ans.values.flatten().tolist()

def most_5(year):
    males=[]
    females=[]
    tmp = names.groupby(['year','gender'])
    for index,group in tmp:
        if(group['gender'].iloc[0] == 'M'):   
            males.append(group.values[:5])
        else:
            females.append(group.values[:5])

    names_list_m = []
    num_list_m =[]
    names_list_f = []
    num_list_f =[]

    for item in males[year-1880]:
        names_list_m.append(item[0])
        num_list_m.append(item[2])


    for item in females[year-1880]:
        names_list_f.append(item[0])
        num_list_f.append(item[2])
        
    return names_list_m,num_list_m,names_list_f,num_list_f

#电影推荐部分函数
def read_data_movie():
    #读取数据
    global user_item
    global data
    global users_info
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('app01/static/ml_1m/ratings.dat', sep='::', header=None, names=rnames,usecols = [0,1,2],engine='python')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('app01/static/ml_1m/movies.dat', sep='::', header=None, names=mnames,usecols = [0,1],engine='python')
    data = pd.merge(ratings,movies)
    users_info = pd.read_table('app01/static/ml_1m/users.dat', sep='::', header=None, names=['user_id','gender','age','occupation'],usecols = [0,1,2,3],engine='python')
    #print(data)
    #print(users_info)
    # 转换成User-Item矩阵
    user_item = ratings.pivot(index='user_id', columns='movie_id', values='rating')


def common_score(user1,user2):
    dict={}
    bool_array = user_item.loc[user1].notnull() & user_item.loc[user2].notnull()
    vector1 = user_item.loc[user1,bool_array]
    vector2 = user_item.loc[user2,bool_array]
    for i,v in vector1.items():
        score=[]
        score.append(user_item.loc[user1][i])
        score.append(user_item.loc[user2][i])
        #在data中去查找电影名字
        moviename = data[data['movie_id']==i]['title'].iloc[0]
        dict[moviename] = score
    return dict


def cal_simi(user1,user2):
    bool_array = user_item.loc[user1].notnull() & user_item.loc[user2].notnull() # scoi
    vector1 = user_item.loc[user1,bool_array]
    vector2 = user_item.loc[user2,bool_array]
    denominator = (sum(vector1*vector1)*sum(vector2*vector2))**0.5
    if(denominator == 0):
        return 0
    s = sum(vector1*vector2) / denominator
    return s


def most_similar_5(user):
    dict={}
    #noloinoloi
    for i in user_item.index:
        if(i == user):
            continue
        dict[i] = cal_simi(user,i)
    #对字典进行排序
    dict = sorted(dict.items(), key=lambda x: x[1],reverse = True)
    ans = dict[0:5]
    #ansdict  user_id:[id,gender,age,job]
    ansdict={}
    for item in ans:
        info=[]
        u = users_info[users_info['user_id']==item[0]].values[0]
        ansdict[item[0]] = u
    #ansdict里面的数字具体代表什么这里就不写了
    return ansdict

def recommend_movies(user): 
    #读取最受欢迎的电影列表 从上往下遍历， 这样推荐出来的顺序就是按欢迎程度排序的
    polular_movies = pd.read_csv('app01/static/ml_1m/popular_movies.csv')
    recommend_users = most_similar_5(user)
    recommend_array = user_item.loc[user].notnull()
    cur_array = user_item.loc[user].notnull()
    for item in recommend_users:
        recommend_array = recommend_array | user_item.loc[item].notnull()

    #我TM傻了
    recommend_array = recommend_array & user_item.loc[user].isnull()#去除自己已经打过分的电影
    #print(recommend_array)
    recommend_list=[]
    for index,row in polular_movies.iterrows():
        # l = []
        # for item in recommend_users:
        #     l.append(user_item.loc[item][row['movieid']])
        # print(row['movieid'])
        # print(l)
        if(cur_array[row['movieid']] == False  and  recommend_array[row['movieid']] == True):
            #要推荐

            moviename = data[data['movie_id']==row['movieid']]['title'].iloc[0]
            recommend_list.append(moviename)

    return recommend_list,recommend_users



#下面是网页相关的函数
def gotoindex(request):
    global names
    if(names is None):
        read_data()
    print(names)
    return render(request,'index.html')



def goto_b1(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        name = request.POST.get('name')
        year = int(request.POST.get('year'))
        name_list =[]
        year_list = []
        num_list = []
        ans = get_births(name,year)
        name_list.append(name)
        year_list.append(year)
        num_list.append(ans)
        print(name_list)
        print(year_list)
        print(num_list)
        return render(request,'babynames_1.html',{'name':name_list,'year':year_list,'num':num_list})
    else:
        return render(request,'babynames_1.html')



def goto_b2(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        name = request.POST.get('name')
        start_year = int(request.POST.get('start_year'))
        end_year = int(request.POST.get('end_year'))
        name_list =[]
        year_list = []
        num_list = []
        name_list.append(name)
        year_list,num_list=get_births_years(name,start_year,end_year)
        print(name_list)
        print(year_list)
        print(num_list)
        return render(request,'babynames_2.html',{'name':name_list,'year':year_list,'num':num_list})
    else:
        print("this is not post!\n")
        return render(request,'babynames_2.html')

def goto_b3(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        name = request.POST.get('name')
        start_year = int(request.POST.get('start_year'))
        end_year = int(request.POST.get('end_year'))
        name_list =name.split(' ')
        num_list =multi_person_births(name_list,start_year,end_year)
        return render(request,'babynames_3.html',{'name':name_list,'start_year':start_year,'end_year':end_year,'num':num_list})
    else:
        print("this is not post!\n")
        return render(request,'babynames_3.html')

def goto_b4(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        name = request.POST.get('name')
        start_year = int(request.POST.get('start_year'))
        end_year = int(request.POST.get('end_year'))
        lifespan = int(request.POST.get('lifespan'))
        name_list =[]
        year_list = []
        num_list = []
        name_list.append(name)
        year_list,num_list = survivor(name,start_year,end_year,lifespan)
        print(type(year_list))
        print(type(num_list))
        return render(request,'babynames_4.html',{'name':name_list,'year':year_list,'num':num_list})
    else:
        print("this is not post!\n")
        return render(request,'babynames_4.html')


def goto_b5(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        namea = request.POST.get('namea')
        nameb = request.POST.get('nameb')
        start_year = int(request.POST.get('start_year'))
        end_year = int(request.POST.get('end_year'))
        namea_list =[]
        nameb_list =[]
        year_list = []
        corre_list =[]
        #name_list.append(nameb)
        # for i in range(start_year,end_year+1):
        #     year_list.append(i)
        num_lista,num_listb,correlation,year_list= Correlation(namea,nameb,start_year,end_year)
        correlation=("%.5f" % correlation)
        corre_list.append(correlation)
        #name 一定要转成list传出去
        namea_list.append(namea)
        nameb_list.append(nameb)
        print(len(year_list))
        print(len(num_lista))
        print(len(num_listb))
        return render(request,'babynames_5.html',{'namea':namea_list,'nameb':nameb_list,'year':year_list,'numa':num_lista,'numb':num_listb,'correlation':corre_list})
    else:
        print("this is not post!\n")
        return render(request,'babynames_5.html')


def goto_b6(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        year = int(request.POST.get('year'))
        name_list =[]
        year_list = []
        num_list = []
        year_list.append(year)
        name_list,num_list = births_oneyear_sort(year)
        return render(request,'babynames_6.html',{'name':name_list[0:5],'year':year_list,'num':num_list[0:5]})
    else:
        return render(request,'babynames_6.html')
    
def goto_b7(request):
    global names
    if(names is None):
        read_data()
    if(request.POST):
        year = int(request.POST.get('year'))
        year_list = []
        name_list_m =[] 
        num_list_m = []
        name_list_f = []
        num_list_f = []
        year_list.append(year)
        name_list_m,num_list_m,name_list_f,num_list_f = most_5(year)
        return render(request,'babynames_7.html',{'name_m':name_list_m,'year':year_list,'num_m':num_list_m,'name_f':name_list_f,'num_f':num_list_f})
    else:
        return render(request,'babynames_7.html')

def goto_movie1(request):
    global data
    global users_info
    global user_item
    #if(data == None or users_info == None or user_item == None):
    #每次进入页面都要重新读取电影文件（好慢= =）
    read_data_movie()
    if(request.POST):
        dic ={}
        #从页面获取评分结果并写入字典
        if(request.POST.get('2858') != 'null' and request.POST.get('2858') !=None):
            dic[2858] = int(request.POST.get('2858'))
        if(request.POST.get('260') != 'null' and request.POST.get('260') !=None):
            dic[260] = int(request.POST.get('260'))
        if(request.POST.get('1196') != 'null' and request.POST.get('1196') !=None):
            dic[1196] = int(request.POST.get('1196'))
        if(request.POST.get('1210') != 'null' and request.POST.get('1210') !=None):
            dic[1210] = int(request.POST.get('1210'))
        if(request.POST.get('480') != 'null' and request.POST.get('480') != None):
            dic[480] = int(request.POST.get('480'))
        if(request.POST.get('2028') != 'null' and request.POST.get('2028') !=None):
            dic[2028] = int(request.POST.get('2028'))
        if(request.POST.get('589') != 'null' and request.POST.get('589') != None):
            dic[589] = int(request.POST.get('589'))
        if(request.POST.get('2571') != 'null' and request.POST.get('2571') !=None):
            dic[2571] = int(request.POST.get('2571'))
        if(request.POST.get('1270') != 'null' and request.POST.get('1270') !=None):
            dic[1270] = int(request.POST.get('1270'))
        if(request.POST.get('593') != 'null' and request.POST.get('593') != None):
            dic[593] = int(request.POST.get('593'))
        if(request.POST.get('1580') != 'null' and request.POST.get('1580') !=None):
            dic[1580] = int(request.POST.get('1580'))
        if(request.POST.get('1198') != 'null' and request.POST.get('1198') !=None):
            dic[1198] = int(request.POST.get('1198'))
        if(request.POST.get('608') != 'null' and request.POST.get('608') != None):
            dic[608] = int(request.POST.get('608'))
        if(request.POST.get('2762') != 'null' and request.POST.get('2762') !=None):
            dic[2762] = int(request.POST.get('2762'))
        if(request.POST.get('110') != 'null' and request.POST.get('110') != None):
            dic[110] = int(request.POST.get('110'))
        if(request.POST.get('2396') != 'null' and request.POST.get('2396')!=None):
            dic[2396] = int(request.POST.get('2396'))
        if(request.POST.get('527') != 'null' and request.POST.get('527') != None):
            dic[527] = int(request.POST.get('527'))
        if(request.POST.get('1617') != 'null' and request.POST.get('1617') !=None):
            dic[1617] = int(request.POST.get('1617'))
        if(request.POST.get('1265') != 'null' and request.POST.get('1265') !=None):
            dic[1265] = int(request.POST.get('1265'))
        if(request.POST.get('1097') != 'null' and request.POST.get('1097') !=None):
            dic[1097] = int(request.POST.get('1097'))
        if(request.POST.get('318') != 'null' and request.POST.get('318') !=  None):
            dic[318] = int(request.POST.get('318'))
        if(request.POST.get('858') != 'null' and request.POST.get('858') !=  None):
            dic[858] = int(request.POST.get('858'))
        if(request.POST.get('356') != 'null' and request.POST.get('356') != None):
            dic[356] = int(request.POST.get('356'))
        if(request.POST.get('296') != 'null' and request.POST.get('296') != None):
            dic[296] = int(request.POST.get('296'))  

        print(dic) 
        #新增加一个用户，加入到user_item列表里
        new = pd.DataFrame(dic,index=[6041])
        user_item = user_item.append(new,ignore_index=False)
        #print(user_item)
        #将用户输入的信息加入了user_item 下面开始推荐电影
        r_movies,r_users = recommend_movies(6041)

        print(r_movies[0:30])
        print(r_users)
        r_users_list =[]
        for index in r_users:
            r_users_list.append(index)
        user1_commonscore=[]
        user2_commonscore=[]
        user3_commonscore=[]
        user4_commonscore=[]
        user5_commonscore=[]
        common_score_list =[]

        #算五个共同评分电影，加入列表（这是个中间变量）
        for key in r_users:
            common_score_list.append(common_score(6041,key))
        #为5个相似用户与当前用户得出相同评分电影，加入各自的列表
        for key in common_score_list[0]:
            dic={}
            dic['moviename'] = key
            dic['myscore'] = common_score_list[0][key][0]
            dic['userscore'] = common_score_list[0][key][1]
            user1_commonscore.append(dic)
        
        for key in common_score_list[1]:
            dic={}
            dic['moviename'] = key
            dic['myscore'] = common_score_list[1][key][0]
            dic['userscore'] = common_score_list[1][key][1]
            user2_commonscore.append(dic)

        for key in common_score_list[2]:
            dic={}
            dic['moviename'] = key
            dic['myscore'] = common_score_list[2][key][0]
            dic['userscore'] = common_score_list[2][key][1]
            user3_commonscore.append(dic)

        for key in common_score_list[3]:
            dic={}
            dic['moviename'] = key
            dic['myscore'] = common_score_list[3][key][0]
            dic['userscore'] = common_score_list[3][key][1]
            user4_commonscore.append(dic)

        for key in common_score_list[4]:
            dic={}
            dic['moviename'] = key
            dic['myscore'] = common_score_list[4][key][0]
            dic['userscore'] = common_score_list[4][key][1]
            user5_commonscore.append(dic)


        print(common_score_list)
        print(user1_commonscore)
        print(user2_commonscore)
        print(user3_commonscore)
        print(user4_commonscore)
        print(user5_commonscore)

        return render(request,'MovieResult.html',{
        'movies':r_movies[0:30],
        'user1_commonscore':user1_commonscore,
        'user2_commonscore':user2_commonscore,
        'user3_commonscore':user3_commonscore,
        'user4_commonscore':user4_commonscore,
        'user5_commonscore':user5_commonscore,
        'user_list':r_users_list,
        'user1id':r_users_list[0], #还要拿出来传一遍，草
        'user2id':r_users_list[1],
        'user3id':r_users_list[2],
        'user4id':r_users_list[3],
        'user5id':r_users_list[4],
        })
    return render(request,'MovieRecommend_1.html')