import simpy as sim
import numpy
import random
import datetime
import statistics
from matplotlib import pyplot as plt

env = sim.Environment()

# Table 1
# Arrival intensity
# time                  = [0,1,2,3,4,5  ,6  ,7  ,8 ,9 ,10,11 ,12 ,13 ,14 ,15,16,17,18,19,20 ,21 ,22 ,23]
arrivalIntensityIndexed = numpy.array([0,0,0,0,0,120,120,120,30,30,30,150,150,150,150,30,30,30,30,30,120,120,120,120])*0.5

SIM_TIME = 60*60*24*365
T_guard = 60 #seconds
P_delay = 0.5 # probability
sInAnHour = 60*60

interarrival_times = []
arrival_times = []

def print_regular_plot(ymin, ymax, y, ylabel, xlabel, x, xmax):
    plt.figure("",figsize=(12,3))
    plt.plot(x,y)
    plt.xticks(numpy.arange(0,xmax+1,1))
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.ylim(ymin,ymax)
    plt.grid()
    plt.show()

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
        delay = random.gammavariate(3.0, X_delay_expected) if delayed > P_delay and X_delay_expected > 0 else 0
        planeArrivalTime = max(numpy.random.exponential(getCurrentArrivalIntensity(env.now)),T_guard) + delay
        interarrival_times.append(round(planeArrivalTime, 1))
        arrival_times.append(env.now)
        yield env.timeout(int(planeArrivalTime))


""" plane_info = [] #landed time, takeoff time
def Plane(env,delay ):
    # delay handled in generator
    # wait for landing
    # print landed time
    # wait for turn around erlang(7, expected = 45*60)
    # request takeoff
    # take-off finished
    # print all variables

# info = {'arrival': 0}

def Runaway(env):
    while True:
        # priority queue and wait for next plane. Takeoff #1 pri
        # hold for T_landing or T_takeoff
        yield env.timeout(delay) """


""" gen = PlaneGen(env)
env.process(gen)
#start two runaway processes?
env.run(until=SIM_TIME) """

def calculate_statistics(results, minTime, maxTime):
    # Iterates over the results from the simulation in order to find the proper population to examine
    population = []
    # print("Min:", minTime, "Max:", maxTime)
    for i in range(len(results[1])):
        # Making the data go in the correct bin regardless of which day it is
        if getClockCurrentDay(results[1][i])*3600 >= maxTime:
            break
        elif getClockCurrentDay(results[1][i])*3600 >= minTime:
            population.append(results[0][i])
            # print(results[0][i])
    # Returns the mean and population standard deviation for the population
    if len(population) < 1 or minTime == 0:
        return 0, 0
    return statistics.mean(population), statistics.pstdev(population)

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

def print_bar_diagram(means, standard_deviations):
    length = numpy.arange(24)
    # Labels for all the different bins
    labels = [
        '00:00', '01:00', '02:00','03:00', 
        '04:00', '05:00', '06:00','07:00', 
        '08:00', '09:00', '10:00','11:00', 
        '12:00', '13:00', '14:00','15:00', 
        '16:00', '17:00', '18:00','19:00', 
        '20:00', '21:00', '22:00','23:00']
    # The following code should create a bar diagram with error margins.
    fig, ax = plt.subplots()
    ax.bar(length, means, yerr=standard_deviations, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Mean Inter-arrival Time [s]')
    ax.set_xticks(length)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

def run_simulation():
    x_values = []
    y_values = []
    means = []
    stds = []
    delays = [0, 10, 30, 60]
    labels = ["μ_delay = {delay} s".format(delay = delays[0]), "μ_delay = {delay} s".format(delay = delays[1]), "μ_delay = {delay} s".format(delay = delays[2]), "μ_delay = {delay} s".format(delay = delays[3])]
    # Code has been refactored to remove the loop. The functions should be able to handle more days without problems
    for i in range(4):
        env = sim.Environment()
        gen = PlaneGen(env, delays[i])
        env.process(gen)
        env.run(until=SIM_TIME)
        results = [interarrival_times.copy(), arrival_times.copy()] 
        x_values.append(arrival_times.copy())
        y_values.append(interarrival_times.copy())
        mean, stddev = calculate_intervals(results)
        means.append(mean)
        stds.append(stddev)
        interarrival_times.clear()
        arrival_times.clear()
        #print_bar_diagram(mean, stddev)
    #print(means)
    multiplot_bar(means, stds, labels, "Arrival time", "Time between arrival and landing [s]", "Time between arrival and landing by arrival time")
    #multiplot(x_values, y_values, labels, "Arrival time", "Time between arrival and landing [s]", "Time between arrival and landing by arrival time")
    return 0


def multiplot(x_values, y_values, labels, x_label, y_label, title):
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

    plt.show()#savefig("lab2/plots/allplots.png", dpi=500)

def multiplot_bar(means, stds, labels, x_label, y_label, title):
    length = numpy.arange(24)
    colors = ["b", "g", "r", "c", "m", "y"]
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
    for row in axs:
        for col in row:
            col.set_xticks(length) 

    for i in range(2):
        for j in range(2):
            axs[i,j].bar(length, means[2*i+j], yerr=stds[2*i+j], align='center', alpha=0.65, capsize=10, color = colors[2*i+j])
            axs[i,j].set_title(labels[2*i+j])
    
    for ax in axs.flat:
        ax.set(ylabel=y_label)

    plt.show()#savefig("lab2/plots/allplots.png", dpi=500)
#print(interarrival_times)

run_simulation()
# print_regular_plot(60, max(interarrival_times)+50, interarrival_times, "Interarrival Time [s]", "Hour", numpy.array(arrival_times)/3600, SIM_TIME/3600)
"""
Spørsmål:


"""
""" x_values = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
y1 = [1,2,3,4,5]
y2 = [5,4,3,2,1]
y3 = [2,4,6,8,10]
y4 = [10,8,6,4,2]
y_values = [y1,y2,y3,y4]
multiplot(x_values, y_values, ["1", "2", "3", "4"], "x", "y", "title") """