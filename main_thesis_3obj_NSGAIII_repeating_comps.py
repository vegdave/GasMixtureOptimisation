import random
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from unicodedata import category
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from csv import reader

from deap import base
from deap import creator
from deap import tools

from math import comb

from itertools import combinations, permutations

from numpy.core.numeric import ones
from scipy.stats import gaussian_kde

from prop_estimation.ate import run_ate
from prop_estimation.GWPcalc import run_gwp
from prop_estimation.electric_estimate import en_estimate_batch
from prop_estimation.boiling_estimate import boiling_estimate_linear, boiling_max, boiling_bezier_batch

from timeit import default_timer as timer


IND_NUMBER = 500   # Starting number of individuals in the population   
IND_SIZE = 3        # Number of gases in the mixture
ITER_NUM = 30   # Number of generations

ALL_COMBO = True    # Consider all mixture ratios

# Genereal container for gases 
class Compound:
    def __init__(self, name, weight, gwp, toxic, en_crit, boiling_point):
        self.name = name
        self.en_crit = float(en_crit)
        self.gwp = float(gwp)
        self.weight = float(weight)
        self.toxic = float(toxic)      # If unknown = -1
        self.boiling_point = float(boiling_point)
        self.fitness = 1
        self.category = -1

gases_list = []
flag = False

# Extracting compounds into containers
df = pd.read_csv(r'data//CompleteGasData_Confirmed_CSV.csv')
#df = df[(df.flammable != 'Yes') & (df.flammable != 'Medium')] 
print("Total of gases: " + str(len(df)))
df = df[df.flammable != 'Yes']  # Removing flammable gases

params_select = ['gas', 'mol_weight', 'gwp', 'toxicity', 'crit_en',	'boiling_point']
data_selection = df.loc[:, params_select]
data_selection = data_selection.dropna()
data_selection = data_selection.reset_index(drop=True)
print("Numebr of gases to be used in optimisation: " + str(len(data_selection)))

gas_labels = data_selection.loc[:, 'gas']

for i in range(len(data_selection)):
    gases_list.append(Compound(data_selection.loc[i, 'gas'], data_selection.loc[i, 'mol_weight'], 
                               data_selection.loc[i, 'gwp'], data_selection.loc[i, 'toxicity'], 
                               data_selection.loc[i, 'crit_en'], data_selection.loc[i, 'boiling_point']))

# Data clustering
data_selection = data_selection.iloc[:, 2:]     # Currently using GWP, ATE, EN, Boiling Point

pt = MinMaxScaler()
pt.fit(data_selection)
data_selection = pt.transform(data_selection)

data_scaled = pd.DataFrame(data_selection)
data_scaled.columns = params_select[2:]

# SHOW DENDROGRAM
#plt.figure(figsize=(10, 7))  
#plt.title("Dendrogram")  
#dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
#plt.savefig("dendogram.png", format='png', dpi=150)
#plt.show()

category_number = 5
cluster = AgglomerativeClustering(n_clusters=category_number, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)

# Visualizing clusters in 3D
data_scaled['gas'] = gas_labels

data_scaled = data_scaled.rename(columns={'gwp': 'GWP', 'crit_en': 'E/Ncrit', 'boiling_point': 'Boiling point'})

if len(data_scaled.columns) >= 3:
    x_ax = params_select[2]
    y_ax = params_select[4]
    z_ax = params_select[5]

symbols = {0:'diamond', 1:'square', 2:'diamond-open', 3:'circle', 4:'circle-open', 5:'square-open'}
#symbols = ['circle-open', 'circle', 'circle-open-dot', 'square', 'triangle', 'diamond-open']
fig = px.scatter_3d(data_scaled, x='GWP', y='E/Ncrit', z='Boiling point', color=cluster.labels_, title=str(params_select[2:]), 
                    color_continuous_scale=px.colors.qualitative.Prism, symbol=cluster.labels_, 
                    symbol_sequence=symbols, text='gas')
#fig.write_html("clustering_unlabelled.html")
#fig.show()

# Assign cluster predictions to gases and calculate range/average of cluster values
en_averages = np.zeros(category_number)
bp_averages = np.zeros(category_number)
gwp_averages = np.zeros(category_number)
tox_averages = np.zeros(category_number)

en_min, en_max = np.full(category_number, -1), np.full(category_number, -1)
bp_min, bp_max = np.full(category_number, -1), np.full(category_number, -1)
gwp_min, gwp_max = np.full(category_number, -1), np.full(category_number, -1)
tox_min, tox_max = np.full(category_number, -1), np.full(category_number, -1)

total_el_in_category = np.zeros(category_number)

# Calculating cluster avg values
for i in range(len(gases_list)):
    gases_list[i].category = cluster.labels_[i]

    en_averages[cluster.labels_[i]] += gases_list[i].en_crit
    if en_min[cluster.labels_[i]] > gases_list[i].en_crit or en_min[cluster.labels_[i]] == -1:
        en_min[cluster.labels_[i]] = gases_list[i].en_crit
    if en_max[cluster.labels_[i]] < gases_list[i].en_crit or en_max[cluster.labels_[i]] == -1:
        en_max[cluster.labels_[i]] = gases_list[i].en_crit

    bp_averages[cluster.labels_[i]] += gases_list[i].boiling_point
    if bp_min[cluster.labels_[i]] > gases_list[i].boiling_point or bp_min[cluster.labels_[i]] == -1:
        bp_min[cluster.labels_[i]] = gases_list[i].boiling_point
    if bp_max[cluster.labels_[i]] < gases_list[i].boiling_point or bp_max[cluster.labels_[i]] == -1:
        bp_max[cluster.labels_[i]] = gases_list[i].boiling_point

    gwp_averages[cluster.labels_[i]] += gases_list[i].gwp
    if gwp_min[cluster.labels_[i]] > gases_list[i].gwp or gwp_min[cluster.labels_[i]] == -1:
        gwp_min[cluster.labels_[i]] = gases_list[i].gwp
    if gwp_max[cluster.labels_[i]] < gases_list[i].gwp or gwp_max[cluster.labels_[i]] == -1:
        gwp_max[cluster.labels_[i]] = gases_list[i].gwp

    tox_averages[cluster.labels_[i]] += gases_list[i].toxic
    if tox_min[cluster.labels_[i]] > gases_list[i].toxic or tox_min[cluster.labels_[i]] == -1:
        tox_min[cluster.labels_[i]] = gases_list[i].toxic
    if tox_max[cluster.labels_[i]] < gases_list[i].toxic or tox_max[cluster.labels_[i]] == -1:
        tox_max[cluster.labels_[i]] = gases_list[i].toxic

    total_el_in_category[cluster.labels_[i]] += 1

for i in range(category_number):
    en_averages[i] /= total_el_in_category[i]
    bp_averages[i] /= total_el_in_category[i]
    gwp_averages[i] /= total_el_in_category[i]
    tox_averages[i] /= total_el_in_category[i]

# Assigning clusters to three gas slots
mixture_slots = [[], [], []]

print('EN avg: ' + str(en_averages))
print('bp avg: ' + str(bp_averages))
print('gwp avg: ' + str(gwp_averages))
print('tox avg: ' + str(tox_averages))

en_q25, en_q50, en_q75 = np.quantile(en_averages, [0.25, 0.5, 0.75])
gwp_q25, gwp_q50, gwp_q75 = np.quantile(gwp_averages, [0.25, 0.5, 0.75])
bp_q25, bp_q50, bp_q75 = np.quantile(bp_averages, [0.25, 0.5, 0.75])

for i in range(category_number):                   
    if en_averages[i] >= en_q50:
        mixture_slots[0].append(i)
    if en_averages[i] >= en_q25 or gwp_averages[i] <= gwp_q25:
        mixture_slots[1].append(i)
    if bp_averages[i] <= en_q25 and gwp_averages[i] <= gwp_q25:
        mixture_slots[2].append(i)

mixture_slots = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
print(mixture_slots)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))       # Objectives: GWP, e/n, Dew point
creator.create("FitnessSingle", base.Fitness, weights=(1.0,))              # Mixtures Fitness as a sum of most popular gases
creator.create("Individual", np.ndarray, fitness=creator.FitnessSingle)
creator.create("Sub_Individual", list, fitness=creator.FitnessMulti)     

# Return a random mixture of compounds with rules for gas slots
def generate_rand_mixture():
    mixture = [[],[],[]]
    i = 0
    while i < IND_SIZE:
        el = random.choice(gases_list)
        if el.category in mixture_slots[i]:
            mixture[i] = el
            i += 1
    #mixture = random.sample(gases_list, IND_SIZE)
    return tuple(mixture)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual,
                generate_rand_mixture, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Return the compound names of the given mixture
def mixture_id(ind):
    #print("mixture contains:")
    name = ''
    for data in ind:
        for compound in data:
            name += compound.name + ','
            #print(compound.name)
        name = name[:-1]
    return name

# Genereate an array of fractions for gas mixtures
def gen_fractions():
    step = 5            # Select a step between the ratios
    arr_fractions = []
    fraction = [0 for i in range(IND_SIZE)]
    # Skip fractions that are too small
    if IND_SIZE == 3:
        if step > 3:
            fraction[0] = 100 - 2 * step
            fraction[1] = step
            fraction[2] = step   
        else:
            fraction[0] = 94
            fraction[1] = 3
            fraction[2] = 3 
    elif IND_SIZE == 2:
        if step > 3:
            fraction[0] = 100 - step
            fraction[1] = step     
        else:
            fraction[0] = 98
            fraction[1] = 2

    #arr_fractions.append(fraction.copy())       100 percent is removed to skip single gases check
    for i in range(fraction[0]//step):
        fraction[0] -= step
        fraction[i % (len(fraction) - 1) + 1] += step
        for element in permutations(fraction):
            if list(element) not in arr_fractions and (0 not in element): # Check for duplicates and dismiss mixtures with zeros
                arr_fractions.append(list(element))

    return arr_fractions


# Generating a sub-population of different gas fractions in every mixture with the calculated fitness
def create_sub_pop(pop, fractions):
    sub_pop = []

    for ind in pop:
        for mix in ind:
            name = mixture_id(ind)
            gwp = run_gwp(mix, fractions)
            ate = run_ate(IND_SIZE, mix, fractions)
            en = en_estimate_batch(mix, fractions)
            boiling = boiling_bezier_batch(mix, fractions)  
            #bde = bde_to_fit(ind)
            for i in range(len(fractions)):
                if ate[i] > 4500:
                    sub_pop.append(creator.Sub_Individual([name, tuple(fractions[i].copy())]))
                    sub_pop[-1].fitness.values = gwp[i], -en[i], boiling[i]

    print("Number of generated mixtures in sub pop:" + str(len(sub_pop)))
    return sub_pop


# Choosing a parent from a population using roulette wheel selection
def roulette_wheel(pop):
    total_fit = 0
    parents = []

    for ind in pop:
        for value in ind.fitness.values:
            total_fit += value

    for i in range(2):
        pick = random.uniform(0, total_fit)
        current = 0
        for ind in pop:
            for value in ind.fitness.values:
                current += value
            if current > pick:
                parents.append(ind)
                break
    return parents


# Swapping one random gas in the mixtures 
def crossover(parents):
    child1, child2 = [toolbox.clone(ind) for ind in (parents)]

    while True:
        pointer1 = random.randint(0, IND_SIZE-1)
        pointer2 = random.randint(0, IND_SIZE-1)

        # Checking slot compatability
        if child1[0][pointer1].category in mixture_slots[pointer2] and child2[0][pointer2].category in mixture_slots[pointer1]:

            tmp = child1[0][pointer1]
            child1[0][pointer1] = child2[0][pointer2]
            child2[0][pointer2] = tmp
            del child1.fitness.values
            del child2.fitness.values

            return child1, child2

  
# Checking how many times a compound appers in the pareto front
# and incrementing the fitness with each match
def compound_evaluation(pareto): 
    for ind in pareto:
        names = ind[0].split(',')
        previous_compounds = []
        for i in range(len(names)):
            no_match = True
            j = 0
            while (j < len(gases_list)) and no_match:
                if (names[i] not in previous_compounds):                    
                    if (names[i] == gases_list[j].name):
                        gases_list[j].fitness += 1        
                        previous_compounds.append(names[i])
                        no_match = False
                else: 
                    no_match = False
                j += 1
                    #no_gas_error = False
    


# Reset the all gas fitnesses to initial values
def reset_gases():
    for gas in gases_list:
        gas.fitness = 1


# Check for duplicates in the population
# Returns a list of repeating mixtures
def mixtures_are_unique(pop):
    duplicates = []
    for a, b in combinations(pop, 2):
        if np.array_equal(a, b):
            duplicates.append(mixture_id(a))
    #dup_dict = {i:duplicates.count(i) for i in duplicates}   # Counting duplicates and creating a dictionary

    return duplicates


# Mixture fitness is the sum of all compound fitnesses
def mixture_evaluation(pop):
    for ind in pop:
        sum = 0
        for mix in ind:
            repeating_comp = []
            count = 0
            for compound in mix:
                if compound.name not in repeating_comp:
                    sum += compound.fitness
                    count += 1
                    repeating_comp.append(compound.name)
        if len(repeating_comp) == 1: 
            ind.fitness.values = 0,      # if individual is a single gas
        else:
            if sum > 0:
                ind.fitness.values = sum / count,
            else:
                ind.fitness.values = 0,
        #print(ind.fitness.values)


# Returns the popularity of compounds in the population
def count_gases(pop):
    names = []
    for ind in pop:
        names.extend(mixture_id(ind).split(','))

    counted = {i:names.count(i) for i in names}

    # Sorting the output in descending order
    return dict(sorted(counted.items(), key=lambda item: item[1], reverse=True))
            

def plot_pop(gen_num, pop, title_txt):
    gen_arr = []
    group = 0
    for el in pop:
        gwp = el.fitness.values[0]
        en = el.fitness.values[1]
        bp = el.fitness.values[2]

        gen_dict = {}
        combined_fitness = -(gwp / 25500) + (-en / 972) - (bp / 300) + 1     # For color coding not used in the algorithm
        gen_dict.update([('Group', group), ('Name', el[0]), ('GWP', gwp), ('E/Ncrit (Td)', -en), 
                         ('Dew Point (K)', bp), ('Ratio', el[1]), ('Fitness', combined_fitness)])
        gen_arr.append(gen_dict)

    gen_df = pd.DataFrame(gen_arr)

    # Calculate point density for color coding
    xyz = np.vstack([gen_df['GWP'], gen_df['Dew Point (K)'], gen_df['E/Ncrit (Td)']])
    density = gaussian_kde(xyz)(xyz)
    gen_df['Density'] = density

    #px.colors.qualitative.Plotly
    gen_plot = px.scatter_3d(gen_df, x = 'GWP', y = 'Dew Point (K)', z = 'E/Ncrit (Td)', color_continuous_scale= px.colors.sequential.Turbo,
                             color= 'Density', hover_data= gen_df[['Name', 'Ratio']], title = title_txt)
    gen_plot.update_traces(marker=dict(size=4), showlegend=False)
    gen_plot.write_html("plots/" + title_txt + ".html")
    camera = dict(eye=dict(x=2, y=-2, z=1))
    gen_plot.update_layout(scene_camera=camera,
                           scene = dict (yaxis = dict(nticks=9, range=[180, 280],),
                                         xaxis = dict(nticks=7, range=[-10, 1300],),
                                         zaxis = dict(nticks=6, range=[-10, 600],)))
    #gen_plot.show()


# Plotting the statistics of the evolution
def plot_stats(logbook, num=0):
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    std = logbook.select("std")
    minimum = logbook.select("min")
    maximum = logbook.select("max")

    avg_fit = [[],[],[]]
    std_fit = [[],[],[]]
    min_fit = [[],[],[]]
    max_fit = [[],[],[]]

    # Extracting feature data from the logbook
    for i in range(len(gen)):
        for j in range(3):
            if j == 1:              # Invert the E/Ncrit axis
                avg_fit[j].append(-avg[i][j])
                std_fit[j].append(std[i][j])
                min_fit[j].append(-minimum[i][j])
                max_fit[j].append(-maximum[i][j])
            else:
                avg_fit[j].append(avg[i][j])
                std_fit[j].append(std[i][j])
                min_fit[j].append(minimum[i][j])
                max_fit[j].append(maximum[i][j])
    
    # Plotting the data
    _, axis = plt.subplots(3, 1, figsize=(12, 10))
    for i in range(3):
        axis[i].plot(gen, avg_fit[i], "b-", )
        axis[i].set_xlabel("Generation")

        if i == 0:
            y_label = "GWP"
        elif i == 1:
            y_label = "E/Ncrit (Td)"
        else:
            y_label = "Dew Point (K)"
        axis[i].set_ylabel(y_label)
        axis[i].set_xticks(np.arange(0, ITER_NUM+1, ITER_NUM//20))

    plt.savefig('plots/avg_stats' + str(num) + '.png', bbox_inches='tight')

    _, axis = plt.subplots(3, 1, figsize=(12, 10))
    for i in range(3):
        axis[i].plot(gen, std_fit[i], "b-", )
        axis[i].set_xlabel("Generation")

        if i == 0:
            y_label = "GWP"
        elif i == 1:
            y_label = "E/Ncrit (Td)"
        else:
            y_label = "Dew Point (K)"
        axis[i].set_ylabel(y_label)
        axis[i].set_xticks(np.arange(0, ITER_NUM+1, ITER_NUM//20))

    plt.savefig('plots/std_stats' + str(num) + '.png', bbox_inches='tight')

    
def main(f):
    cxchance = 0.6      # Probability of crossover
    mu = 0.01            # Mutation probability

    pop_size = IND_NUMBER   # Current population size

    pop = toolbox.population(n=pop_size)      # Main population of random gas mixtures
    #pop = init_pop()
    
    # Statistics object to store data
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    
    f.write("NSGA III Selection\n")
    f.write("Initial population size: " + str(pop_size) + '\n' + "Number of generations: " 
            + str(ITER_NUM) + '\n' + "Number of gases: " + str(len(gases_list)) 
            + '\n' + "Number of gas slots: " + str(IND_SIZE) + '\n' + 'Gas structure: ' + str(mixture_slots) +'\n')

    print('There are %d mixtures combinations for selected sizes' % comb(len(gases_list), IND_SIZE))

    # Refrerece points for NSGA-III
    NOBJ = 3
    P = [12]
    SCALES = [1]

    ref_points = [tools.uniform_reference_points(NOBJ, p, s) for p, s in zip(P, SCALES)]
    ref_points = np.concatenate(ref_points)
    _, uniques = np.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[uniques]

    resume = False # Variable to stop iterations at 50% iterations

    fractions = gen_fractions()
    # Main generations loop
    for gen_num in range(ITER_NUM):
        print("Iteration:" + str(gen_num+1))

        gas_popularity = count_gases(pop)

        # Add data to the log
        f.write("Iteration: " + str(gen_num+1) + ' ' + str(gas_popularity) + '\n') 
        
        # Creating a sub-population of different gas fractions in every mixture with the calculated fitness
        sub_pop = create_sub_pop(pop, fractions)
        
        # Gathering statistics
        record = stats.compile(sub_pop)
        logbook.record(gen=gen_num, **record)

        # Updating the sub pareto front      
        selection_method = "NSGA3" 
        if len(sub_pop) // 3 < 3000:
            select_size = len(sub_pop) // 3
        else:
            select_size = 3000
        sub_hof = tools.selNSGA3(sub_pop, select_size, ref_points=ref_points)      

        print("Currently %d solutions in the sub pareto front" % len(sub_hof))
    
        compound_evaluation(sub_hof)

        for gas in gases_list:
            f.write(gas.name + ' / ' + str(gas.fitness) + '#')
        f.write('\n')

        mixture_evaluation(pop)

        # Plotting and storing the HOF of the sub-population
        """if len(sub_pop) > 0:
            if (gen_num == ITER_NUM-1 or gen_num == 0):
                plot_pop(gen_num, sub_pop, 'AllSubpopIndividuals_Gen_'+ str(gen_num) )
                #print("__________Plotting___________")
            if (gen_num == ITER_NUM-1 or gen_num == 0):
                plot_pop(gen_num, sub_hof, selection_method + '_Gen_' + str(gen_num))"""

        # Show top mixtures
        pop.sort(key=lambda x: x.fitness.values, reverse=True)
        if pop_size > 100:
            display_size = 100
        else:
            display_size = pop_size
        for i in range(display_size):
            f.write(str(i+1) + ". " + mixture_id(pop[i]) + " " + str(pop[i].fitness.values) + "#")
        f.write('\n\n')

        # Generating offspring from the main population
        children = []
        while len(children) < pop_size:
            mate = random.random() < cxchance 
            mutate = random.random() < mu
            parents = roulette_wheel(pop)
            if mate:                    
                for ch in crossover(parents):
                    if mutate:
                        position = random.randint(0, IND_SIZE-1)
                        mutation = random.choice(gases_list)
                        # Make sure that mutation is different from current gas and fits in the slot
                        while (ch[0][position] == mutation) or (mutation.category not in mixture_slots[position]):
                            mutation = random.choice(gases_list)
                        ch[0][position] = mutation
                        
                    children.append(ch)
            else:
                # If no crossover, just copy the parents
                children.append(parents[0])
                children.append(parents[1])

        pop = children
        
        print('Pop_size = ' + str(pop_size))

        reset_gases()
        
        if (gen_num >= ITER_NUM // 2) and resume:
            input('Enter any value to continue: ')
            resume = False
       
    return pop, logbook, sub_hof 

# Run the program
start = timer()
for i in range (100):
    f = open('logs/evolution_log_' + str(i) + '.txt', "a")  

    pop, logbook, hof = main(f)

    print("Time elapsed: " + str(timer() - start))

    f.write("Time elapsed: " + str(timer() - start) + '\n\n')
    f.close()

    plot_stats(logbook, num=i)