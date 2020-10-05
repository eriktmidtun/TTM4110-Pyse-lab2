import simpy as sim
import numpy
import random
import datetime
import statistics
from matplotlib import pyplot as plt

env = sim.Environment()

# Table 1
# Arrival intensity
# time                               =[0,1,2,3,4,5  ,6  ,7  ,8 ,9 ,10,11 ,12 ,13 ,14 ,15,16,17,18,19,20 ,21 ,22 ,23]
arrivalIntensityIndexed = numpy.array([0,0,0,0,0,120,120,120,30,30,30,150,150,150,150,30,30,30,30,30,120,120,120,120])/3333


DAYS = 30
SIM_TIME = 60*60*24*30
T_guard = 60 #seconds
P_delay = 0.5 # probability
sInAnHour = 60*60

interarrival_times = []
arrival_times = []
schedule_times = []

def getCurrentArrivalIntensity(currenttime):
    clockCurrentDay = getClockCurrentDay(currenttime)
    return arrivalIntensityIndexed[int(clockCurrentDay)]

def getClockCurrentDay(currenttime):
    #currenttime is in seconds
    currentHour = currenttime / sInAnHour
    return currentHour % 24

def PlaneGen(env, X_delay_expected):
    plane = 0
    while True:
        clock = getClockCurrentDay(env.now)
        if clock < 5:
            #interarrival_times.append(T_guard) #fix points on the end of day
            #arrival_times.append(env.now)
            yield env.timeout(5*sInAnHour-clock*sInAnHour) # wait til 05:00
            #fix points on the start of day
            #interarrival_times.append(T_guard)
            #arrival_times.append(env.now)
        plane += 1
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        #Plane(env)
        delayed = random.uniform(0.0,1.0)
        delay = random.gammavariate(3.0, X_delay_expected/3) if delayed > P_delay and X_delay_expected > 0 else 0
        planeArrivalTime = max(numpy.random.exponential(1/getCurrentArrivalIntensity(env.now)),T_guard)
        interarrival_times.append(round(planeArrivalTime, 1))
        schedule_times.append(env.now)
        arrival_times.append(env.now+delay)
        yield env.timeout(int(planeArrivalTime))

def calculate_statistics(results, minTime, maxTime):
    # Iterates over the results from the simulation in order to find the proper population to examine
    population = []
    # print("Min:", minTime, "Max:", maxTime)
    for i in range(len(results[1])):
        # Making the data go in the correct bin regardless of which day it is
        if getClockCurrentDay(results[1][i])*3600 >= maxTime:
            break
        elif getClockCurrentDay(results[1][i])*3600 >= minTime:
            population.append(getClockCurrentDay(results[0][i]))
            # print(results[0][i])
    # Returns the mean and population standard deviation for the population
    if len(population) < 1 or minTime == 0:
        return 0, 0
    return len(population), statistics.stdev(population)

def calculate_intervals(results):
    # We're using 48 buckets as we're creating a new bin every 30 mins.
    number_of_bins = 24
    mean = []
    stddev = []
    for i in range(number_of_bins):
        # Find the mean and standard deviation for the current bin and append to the corresponding arrays
        i_mean, i_std = calculate_statistics(results, 3600*i, 3600*(i+1))
        mean.append(i_mean)
        stddev.append(i_std)
    return mean, stddev

def run_simulation():
    x_values = []
    y_values = []
    means = []
    stds = []
    delays = [0, 300, 600, 900 ]
    labels = ["μ_delay = {delay} s".format(delay = delays[0]), "μ_delay = {delay} s".format(delay = delays[1]), "μ_delay = {delay} s".format(delay = delays[2]), "μ_delay = {delay} s".format(delay = delays[3])]
    # Code has been refactored to remove the loop. The functions should be able to handle more days without problems
    for i in range(4):
        env = sim.Environment()
        gen = PlaneGen(env, delays[i])
        env.process(gen)
        env.run(until=SIM_TIME)
        results = [schedule_times.copy(), arrival_times.copy()] 
        x_values.append(arrival_times.copy())
        y_values.append(interarrival_times.copy())
        mean, stddev = calculate_intervals(results)
        means.append(mean)
        stds.append(stddev)
        schedule_times.clear()
        interarrival_times.clear()
        arrival_times.clear()
    #print(means)
    #multiplot(x_values, y_values, labels, "Arrival time", "Time between arrival and landing [s]", "Time between arrival and landing by arrival time", "a_oneDay")
    multiplot_bar(means, stds, labels, "Arrival time", "Number of planes", "Graph showing correlation between time and number of arrivals", "a")
    return 0


def multiplot(x_values, y_values, labels, x_label, y_label, title, filename):
    colors = ["b", "g", "r", "c", "m", "y"]
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
    for row in axs:
        for col in row:
            col.set_xticks(numpy.arange(5,24,1))

    for i in range(2):
        for j in range(2):
            axs[i,j].plot(getClockCurrentDay(numpy.array(x_values[2*i+j])), y_values[2*i+j], color = colors[2*i+j])
            axs[i,j].set_title(labels[2*i+j])
    
    for ax in axs.flat:
        ax.set(ylabel=y_label)

    plt.suptitle(title, fontsize=25)
    plt.savefig("lab2/plots/" + filename + ".png", dpi=500)

def multiplot_bar(means, stds, labels, x_label, y_label, title, filename):
    length = numpy.arange(24)
    colors = ["b", "g", "r", "c", "m", "y"]
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
    for row in axs:
        for col in row:
            col.set_xticks(length)
            col.grid()

    for i in range(2):
        for j in range(2):
            axs[i,j].bar(length, means[2*i+j], yerr=stds[2*i+j], align='center', alpha=0.65, capsize=10, color = colors[2*i+j])
            axs[i,j].set_title(labels[2*i+j])
    
    for ax in axs.flat:
        ax.set(ylabel=y_label)

    plt.suptitle(title, fontsize=25)
    plt.savefig("lab2/plots/" + filename + ".png", dpi=500)

run_simulation()
