import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def initialPopulation(popSize, numItems):
    population = []
    for _ in range(popSize):
        bits = []
        for _ in range(numItems):
            r = random.random()
            if r < 0.5:
                bits.append(0)
            else:
                bits.append(1)
        population.append(bits)
    return population

def fitness(bits, price, weight, capacity):
    total_price = total_weight = 0
    for i in range(len(bits)):
        if bits[i] == 1:
            total_price += price[i]
            total_weight += weight[i]
    if total_weight > capacity:
        return 0
    return total_price  

def crossOver(parent1, parent2):
    n = len(parent1)
    cut = random.randint(1, n - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return child1, child2

def flipMutation(individual):
    n = len(individual)
    p1 = random.randint(0, n - 1)

    individual2 = individual.copy()
    individual2[p1] = 1 - individual2[p1]
    return individual2

def nextPopulation(population, crossoverRate, mutationRate, price, weight, capacity):
    children = population.copy()
    popSize = len(population)

    cnt = 0
    while cnt < popSize:
        parent1, parent2 = random.sample(population, 2)

        # if fitness(x1, price, weight, capacity) < fitness(x2, price, weight, capacity):
        #     parent1 = x2
        # else:
        #     parent1 = x1

        # if fitness(x3, price, weight, capacity) < fitness(x4, price, weight, capacity):
        #     parent2 = x4
        # else:
        #     parent2 = x3

        if random.random() < crossoverRate:
            offSpring1, offSpring2 = crossOver(parent1, parent2)
            children.append(offSpring1)
            children.append(offSpring2)
            cnt += 2

        if random.random() < mutationRate:
            children.append(flipMutation(parent1))
            children.append(flipMutation(parent2))
            cnt += 2

    children.sort(key = lambda x: fitness(x, price, weight, capacity), reverse = True)
    nextGen = children[0:popSize]

    return nextGen

def greedyAlgorithm1(price, weight, capacity):
    curWeight = 0
    bits = []
    for i in range(len(price)):
        if curWeight + weight[i] <= capacity:
            bits.append(1)
            curWeight += weight[i]
        else:
            bits.append(0)
    return bits

# def greedyAlgorithm2(price, weight, capacity):
#     curWeight = 0
#     bits = []
#     for i in range(len(price) - 1, -1, -1):
#         if curWeight + weight[i] <= capacity:
#             bits.append(1)
#             curWeight += weight[i]
#         else:
#             bits.append(0)
#     return bits

def geneticAlgorithm(pop, price, weight, capacity, crossoverRate, mutationRate, generations, maximumLoop):
    # pop = initialPopulation(popSize, len(price))
    # pop.append(greedyAlgorithm1(price, weight, capacity))
    # pop.append(greedyAlgorithm2(price, weight, capacity))
    pop.sort(key = lambda x: fitness(x, price, weight, capacity), reverse = True)

    progress = []
    progress.append(fitness(pop[0], price, weight, capacity))

    loopnotImprove = 0
    lastWeight = -1
    for _ in range(generations):
        pop = nextPopulation(pop, crossoverRate, mutationRate, price, weight, capacity)
        curWeight = fitness(pop[0], price, weight, capacity)

        if lastWeight == curWeight:
            loopnotImprove += 1
            if loopnotImprove == maximumLoop:
                break
        else:
            lastWeight = curWeight
            loopnotImprove = 0
        
        # print(fitness(pop[0], price, weight, capacity))
        progress.append(curWeight)

    # plt.xlabel('Generations')
    # plt.ylabel('Fitness')
    # plt.plot(progress)
    # plt.show()

    return progress

df = pd.read_csv('knapPI_3_500_1000_1_items.csv')
df.columns = [c.strip() for c in df.columns]
price = df['price']
weight = df['weight']

# print(price)
# print(weight)

population = initialPopulation(100, len(price))
geneticAlgorithm(population, price, weight, capacity=1000000, crossoverRate=0.9, mutationRate=0.1, generations=500, maximumLoop=100)

# print(fitness(answer, price, weight, capacity=10000))

# for i in range(len(answer)):
#     if answer[i] == 1:
#         print(i)