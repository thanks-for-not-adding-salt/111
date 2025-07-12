import pandas as pd
from numpy.random import randint
from numpy.random import rand
import functools
import numpy as np
from patternComponents import LowPoint
from multiprocessing import Pool
from  paramConfig import *

goodList = []
resultRecorder = []
def genetic_algorithm(objective,n_bits,n_iter,n_pop,r_cross,r_mut):
 # 种群初始化
 #pop = [randint(0,2,n_bits).tolist() for_in range(n_pop)]
 #这种种群初始化的方式，随机生成0/1，容易过拟合
   pop = [np.random.choice([True,False],n_bits,p=[0.25,0.75]).tolist() for _ in range(n_pop)]
 #todo:p调参
#初始化最佳解和分数
   best, best_eval =0, objective(pop[0])
   for gen in range(n_iter):
       print(f"gen is {gen}\n")
       with Pool(5) as p:
           scores = p.map(objective,pop)
   temp = pd.DataFrame({"pop": pop, 'score': scores}).drop_duplicates(subset = ['score']).sort_values(ascending = False,by ='score')
   goodList.append(temp[temp['score']]>0.7)
   for i in range (n_pop):
      if scores[i] > best_eval:
          best,best_eval = pop[i],scores[i]
          print(">%d,new best f(%s) = %.3f"% (gen,pop[i],scores[i]))
   selected = [selection(pop,scores) for _ in range(n_pop)]
   children = list()
   for i in range(0,n_pop,2):
       if i+1>= n_pop:
           continue
       p1,p2 = selected[i],selected[i+1]
       for c in crossover (p1,p2,r_cross):
                 mutation(c,r_mut)
                 children.append(c)
   pop = children
   n_pop = len(children)
   return [best,best_eval]
def onemax(x):
    return -sum(x)
def fitness_func(df_results,colsList):
    global fitness_calculate
    def fitness_calculate(solution):
        conditionList = colsList[solution].tolist()
        if len(conditionList) == 0:
            return -100#todo
        else:
            dfs = [df_results[ele] for ele in conditionList]
            if len(dfs) == 1:
                ldld = df_results[dfs[0].tolist()]
            else:
                ldld = df_results[functools.reduce(lambda x,y: x*y, dfs)]
                if ldld.shape[0] >= 15:
                    winrate = (ldld['delta']>0).sum()/ldld.shape[0]
                    averagedWin = ldld['delta'].mean()
                    score = winrate + averagedWin
                else:
                    score =  -100
                return score

    return fitness_calculate


def selection(pop,scores,k=3):
    selection_ix = randint(len(pop))
    for ix in randint(1,len(pop),k-1):
        if scores[ix]>scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover (p1,p2,r_cross):
    pass
    c1,c2 =p1.copy(),p2.copy()
    if len(p1)>3 and rand() < r_cross:
        pt = randint(1,len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1,c2]
def mutation(bitstring,r_mut):

    for i in range(len(bitstring)):
        if rand() < r_mut:
            if rand()<0.30:#todo
                 bitstring[i] = False
            else:
                 bitstring[i] = True if bitstring[i] == False else False
