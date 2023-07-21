import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import random
import statistics
from sklearn.preprocessing import StandardScaler

#Valve closing then opening timings
#input_valves=[(55,5,0),(45,15,0),(55,5,0),(230,10,0),(55,5,0),(55,5,0),(50,10,0),(45,15,0),(50,10,0)]
#input_valves=[(15,45,1),(45,15,1),(40,30,1),(80,20,1),(130,10,1),(20,30,1),(30,10,1),(30,20,1),(35,25,5),(40,10,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1),(30,30,1)]

def valve_overlapping(input_valves,population_size=200,gen_theshold=20):
    valves=[]
    priority=[]
    for i,j,k in input_valves:
        valves.append((i,j))
        priority.append(k)
    total_time=[i+j for (i,j) in valves]
    LCM=math.lcm(*total_time)

    #Create chromosome
    def create_chromosome(valves):
        chromosome=[]
        for j in range(len(valves)):
            index=random.randint(0,1)
            
            shift=random.randint(0,total_time[j])
            gene=([0]*valves[j][0]+[1]*valves[j][1])*int(LCM/total_time[j])
            gene=gene[shift:]+gene[:shift]
        
            chromosome.append(gene)

        return chromosome


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

        sorted_list_pair=sorted(zip(fitness_list,population))
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

            #Elitism 20%
            s=int((20*population_size)/100)
            new_population.extend(population[:s])

            #Mating from 60% of fittest population forming rest of 80% population
            s = int((80*population_size)/100)
            for _ in range(s):
                parent1 = random.choice(population[:60])
                parent2 = random.choice(population[:60])
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
                elif num==1:
                    if count0!=0:
                        break
                    count1+=1
                    if count1==valves[i][1]:
                        break
            
                
            schedule.append(count0+count1)
        
        schedule=[x-min(schedule) for x in schedule]
        return schedule

    
    priority_number=priority_value(best_chromosome)
    overlap_number=sum_timings.count(max(sum_timings))
    #print("LCM is:",LCM)
    schedule=scheduling(best_chromosome)

    def scheduled_valves(sum_timings):
        while sum_timings[0]==0:
            sum_timings=sum_timings[1:]+[sum_timings[0]]

        return sum_timings

    scheduled_sum_timings=scheduled_valves(sum_timings)

    return schedule,[scheduled_sum_timings],overlap_number,priority_number


input_valves=[(55,5,1),(45,15,1)]
valve_overlapping(input_valves=input_valves)
'''
normal_calculation=True

if normal_calculation==False:
    population_size=500
    gen_theshold=100
else:
    population_size=200
    gen_theshold=20

schedule,sum_timings,overlap_number,priority_number=valve_overlapping(input_valves,population_size=population_size,gen_theshold=gen_theshold)

print("Maximum Overlap Value Count:",overlap_number)
print("Priority Value:",priority_number)
print("Valve schedule",schedule)

time=np.arange(0, min(len(sum_timings),241))
valve_numbers=sum_timings[:241]

# Plot the orthogonal step function
fig,ax=plt.subplots(figsize=(12, 4))

for i in range(len(time) - 1):
    ax.plot([time[i], time[i + 1]], [valve_numbers[i], valve_numbers[i]], color='blue')
    ax.plot([time[i + 1], time[i + 1]], [valve_numbers[i], valve_numbers[i + 1]], color='blue')
    x = [time[i], time[i + 1], time[i + 1], time[i]]
    y = [valve_numbers[i], valve_numbers[i], 0, 0]
    ax.fill(x, y, color='lightblue', alpha=0.5)
#ax.plot(sum_timings[:240], linestyle='-', color='black')


y_ticks = range(0, max(sum_timings)+3,1)
x_ticks = range(0, min(len(sum_timings),241)+20,10)
plt.yticks(y_ticks)
plt.xticks(x_ticks)
ax.set_title('Indicative valve overlapping')
ax.set_xlabel('Time in minute')
ax.set_ylabel('Valve overlap count')
ax.grid(color='lightgray', linestyle='--')
plt.show()
'''
