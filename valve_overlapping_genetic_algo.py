import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import random
import statistics
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Valve closing then opening timings
#input_valves=[(55,5,0),(45,15,0),(55,5,0),(230,10,0),(55,5,0),(55,5,0),(50,10,0),(45,15,0),(50,10,0)]
#input_valves=[(15,45,1),(45,15,1),(40,30,1),(80,20,1),(130,10,1),(20,30,1),(30,10,1),(30,20,1),(35,25,5),(40,10,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1)]

def valve_overlapping(input_valves=None,population_size=200,gen_theshold=40,cycles=None ):
    
    valves=[]
    priority=[]
    if cycles!=None: 
        input_valves= [( sum(1 for val in cycle if val == 0),sum(1 for val in cycle if val != 0), 0) for cycle in cycles]
        for i,j,k in input_valves:
            valves.append((i,j))
            priority.append(k)
        total_time=[i+j for (i,j) in valves]
        LCM=math.lcm(*total_time)


    if cycles==None:
        for i,j,k in input_valves:
            valves.append((i,j))
            priority.append(k)
        total_time=[i+j for (i,j) in valves]
        LCM=math.lcm(*total_time)
  
    #New Code start 
    def manual_optimization(input_valves=input_valves,total_time=total_time,LCM=LCM):
        possible=True
        sum_timings_m=[]        
        schedule_m=[]
        if LCM not in total_time:
            possible=False
        #print(possible)    
        for i in total_time:
            if LCM%i!=0:
                possible=False
        #print(possible)           
        open_time=[]
        close_time=[]
        for i,j,k in input_valves:
            open_time.append(j)
            close_time.append(i)
        
        total_injection_time=0
        for i,j in zip(total_time,open_time):
            total_injection_time+=(LCM//i)*j
        
        if total_injection_time>LCM:
            possible=False
        #print(possible) 
        #for i in open_time:
        #    if i>min(close_time):
        #        possible=False
        #print(possible) 
        
        if len(set(input_valves))>6:

            all_permutations=[input_valves]
        else:
            all_permutations = list(itertools.permutations(input_valves))

        if possible==True:
            for perm in all_permutations:
                
                current_time=0
                schedule_m=[]
                sum_timings_m_pre=[]
                sum_timings_m=[0]
                def check_overlap(list1, list2):
                    for val1, val2 in zip(list1, list2):
                        if val1 == 1 and val2 == 1:
                            return True
                    return False
                for i in range(len(perm)):
                    
                    while sum_timings_m[current_time]!=0:
                        current_time+=1
                    
                    #print(input_valves[i][0]+input_valves[i][1])
                    to_append=((perm[i][1]*[1]+(perm[i][0])*[0])*int(LCM/(perm[i][0]+perm[i][1])))
                    to_append_shuffle=to_append[-current_time:]+to_append[:-current_time]

                    while check_overlap(to_append_shuffle,sum_timings_m) and current_time<=len(sum_timings_m):
                        current_time+=1
                        to_append_shuffle=to_append[-current_time:]+to_append[:-current_time]


                    sum_timings_m_pre.append(to_append_shuffle)
                    if sum_timings_m==[0]:
                        sum_timings_m=[]
                    schedule_m.append(current_time)
                    current_time=0#current_time+=input_valves[i][1]
                    
                    df=pd.DataFrame(sum_timings_m_pre)
                    sum_timings_m=df.sum(axis=0).to_list()
                possible=False
                if max(sum_timings_m)==1:
                    possible=True
                #print(possible)
                if possible==True:
                    #schedule_indices = [ [index,value] for index, value in zip( schedule_m,perm)]
                    sorted_schedule_m=[]
                    perm=list(perm)
                    for i in input_valves:
                        ii=perm.index(i)
                        sorted_schedule_m.append(schedule_m[ii])
                        perm[ii]=-math.inf
                    schedule_m=sorted_schedule_m
                    sum_indices = { index:value for index, value in zip(perm, sum_timings_m)}
                    #sum_timings_m = [sum_indices[x] for x in input_valves]

                    break

        return possible,schedule_m,[sum_timings_m],"manual",0
        
    A,B,C,D,E=manual_optimization()
    if A==True:
        return B,C,D,E

    #New code end
    #Create chromosome
    def create_chromosome(valves):
        chromosome=[]
        for j in range(len(valves)):
            index=random.randint(0,1)
            
            shift=random.randint(0,total_time[j])#//(min(valves[j][1],valves[j][0])))*min(valves[j][1],valves[j][0])
            gene=([0]*valves[j][0]+[1]*valves[j][1])*int(LCM/total_time[j])
            gene=gene[shift:]+gene[:shift]
        
            chromosome.append(gene)
        if cycles!=None:
            chromosome=[]
            for gene in cycles:
                shift=random.randint(0,len(cycles[0]))
                gene=gene[shift:]+gene[:shift]
                chromosome.append(gene)
        return chromosome
        
    #chromosome=create_chromosome(valves)
    

    #Generate population
    def generate_population(population_size):
        population=[]
        for i in range(population_size):
            population.append(create_chromosome(valves))

        return population


    #Fitness functions

    def priority_valve(chromosome,gene):
        sum_timings=[sum(column) for column in zip(*chromosome)]
        priority=[]
        for i, j in zip(gene,sum_timings):
            if i==1:
                priority.append(j)
            else:
                priority.append(i)
        if max(priority)==1:
            return 0
        else:
            return sum(priority)



    def fitness_stddev(chromosome):
        sum_timings=[sum(column) for column in zip(*chromosome)]
        fitness=statistics.stdev(sum_timings)
        

        #return max(sum_timings)
        return fitness


    def fitness_priority(chromosome):  #USE CAREFULLY WITH STDDEV FITNESS
        value=0
        for gene,p in zip(chromosome,priority):
            if p==1:
                value+=priority_valve(chromosome,gene)
            else: pass
        return value


    def population_fitness(population): 
        fitness_list1=[]
        fitness_list2=[]
        
        for chromosome in population:
            fitness_list1.append([fitness_stddev(chromosome)])
            fitness_list2.append([fitness_priority(chromosome)])
        
        scaler = StandardScaler()

        fitness_list1=list(itertools.chain(*scaler.fit_transform(fitness_list1).tolist()))
        fitness_list2=list(itertools.chain(*scaler.fit_transform(fitness_list2).tolist()))

        fitness_list=[x+100*y for x, y in zip(fitness_list1,fitness_list2)]
        

        sorted_list_pair=sorted(zip(fitness_list,population))   #zip(fitness_list,population)
        sorted_fitness,sorted_population=zip(*sorted_list_pair)

        return sorted_population

    def overlap_value(chromosome):
        sum_timings=[sum(column) for column in zip(*chromosome)]

        return sum_timings

    def priority_value(chromosome):
        value=0
        for gene,p in zip(chromosome,priority):
            if p==1:
                value+=priority_valve(chromosome,gene)
            else: pass
        return value

    #Mutated Chromosome
    def mate(p1,p2):
        child_chromosome=[]
        for i,j in zip(p1,p2):
            prob=random.random()
            if prob<0.5:
                child_chromosome.append(i)
            else:
                child_chromosome.append(j)
        return child_chromosome


    def genetic_algorithm():
        generation=1
        found=False
        temperature = 10000
        population=generate_population(population_size)
        fitness_list=[]

        while generation<=gen_theshold:
            new_population=[]

            population=population_fitness(population)

            #Elitism 25%
            s=int((25*population_size)/100)
            new_population.extend(population[:s])

            #Mating from 50% of fittest population forming rest of 80% population
            s = int((75*population_size)/100)
            for _ in range(s):
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[:50])
                child = mate(parent1,parent2)
                new_population.append(child)

            population=new_population
            generation+=1
        return population[0]   

    best_chromosome=genetic_algorithm()
    sum_timings=overlap_value(best_chromosome)

    def scheduling(chromosome):
        schedule=[]
        for i in range(len(chromosome)):
            count0=0
            count1=0
            for num in chromosome[i]:
                if num == 0:
                    count0 += 1
                    if count0==valves[i][0]:
                        break
                elif num!=0:
                    if count0!=0:
                        break
                    count1+=1
                    if count1==valves[i][1]:
                        break
            
                
            schedule.append(count0+count1)
        if cycles!=None: return schedule 
        schedule=[x-min(schedule) for x in schedule]
        return schedule

    
    priority_number=priority_value(best_chromosome)
    overlap_number=sum_timings.count(max(sum_timings))
    #print("LCM is:",LCM)
    schedule=scheduling(best_chromosome)
    

    def scheduled_valves(schedule,input_valves):
        sum_timings=[]
        for i in range(len(input_valves)):
            sum_timings.append((schedule[i]*[0]+input_valves[i][1]*[1]+(input_valves[i][0]-schedule[i])*[0])*int(LCM/(input_valves[i][0]+input_valves[i][1])))
        df=pd.DataFrame(sum_timings)
        sum_timings=df.sum(axis=0).to_list()

        if cycles!=None:
            sum_timings=best_chromosome
            #for i in range(len(input_valves)):
            ##    s=schedule[i]
            #   sum_timings.append(best_chromosome[i][s:]+best_chromosome[i][:s])
            #df=pd.DataFrame(sum_timings)
            #sum_timings=df.sum(axis=0).to_list()


        '''
        leading_one=0
        while sum_timings[0]!=0:
            leading_one+=1
        
        one_list=[]
        for i,j,k in input_valves:
            one_list.append(j)

        if leading_one<min(one_list):
            while sum_timings[0]==1:
                sum_timings=sum_timings[1:]+[sum_timings[0]]
            while sum_timings[0]==0:
                sum_timings=sum_timings[1:]+[sum_timings[0]]
        '''
        return sum_timings


    scheduled_sum_timings=scheduled_valves(schedule,input_valves)

    
    
    '''
    def scheduled_valves(best_chromosome,schedule):
        scheduled_timings=[]

        for i in range(len(best_chromosome)):
            gene=best_chromosome[i]
            s=schedule[i]
            gene=[np.concatenate((np.zeros(s),np.ones(sum(gene)),np.zeros(len(gene)-s-sum(gene))))]
            scheduled_timings.append(gene)
        #print(scheduled_timings)
        scheduled_sum_timings=[sum(x) for x in zip(*scheduled_timings)]

        return scheduled_sum_timings
    '''
    

    return schedule,[scheduled_sum_timings],"manual",priority_number


#input_valves=[(55,5,1),(5,55,1)]  #,(25,5,1),(55,5,1),(55,5,1),(55,5,1)

#cycles=[(1,0,0,0,0,0),(1,0,0,0,0,0)]
#print(valve_overlapping(input_valves=input_valves))
#print(valve_overlapping(cycles=cycles))
