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
arrivalIntensityIndexed = [0,0,0,0,0,120,120,120,30,30,30,150,150,150,150,30,30,30,30,30,120,120,120,120]

SIM_TIME = 60*60*24*30
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

def PlaneGen(env):
    plane = 0
    X_delay_expected = 0
    while True:
        clock = getClockCurrentDay(env.now)
        if clock < 5:
            #X_delay_expected += 10
            interarrival_times.append(T_guard) #fix points on the end of day
            arrival_times.append(env.now)
            yield env.timeout(5*sInAnHour-clock*sInAnHour) # wait til 05:00
            #fix points on the start of day
            interarrival_times.append(T_guard)
            arrival_times.append(env.now)
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
        '00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', 
        '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', 
        '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', 
        '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', 
        '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', 
        '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']
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
    # Code has been refactored to remove the loop. The functions should be able to handle more days without problems
    gen = PlaneGen(env)
    env.process(gen)
    env.run(until=SIM_TIME)
    results = [interarrival_times, arrival_times]
    mean, stddev = calculate_intervals(results)
    print_bar_diagram(mean, stddev)
    return 0
        



#print(interarrival_times)

run_simulation()
# print_regular_plot(60, max(interarrival_times)+50, interarrival_times, "Interarrival Time [s]", "Hour", numpy.array(arrival_times)/3600, SIM_TIME/3600)
"""
Spørsmål:


"""