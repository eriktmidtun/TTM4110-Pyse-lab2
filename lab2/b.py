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

SIM_TIME = 60*60*24
sInAnHour = 60*60

T_guard = 60 #seconds
P_delay = 0.5 # probability
T_landing = 60 # seconds
T_takeoff = 60 # seconds
X_turnaround_expected = 45*60 #seconds
N_runways = 2 # max runways

plane_number = 0
interarrival_times = []
arrival_times = []


def getCurrentArrivalIntensity(currenttime):
    clockCurrentDay = getClockCurrentDay(currenttime)
    return arrivalIntensityIndexed[int(clockCurrentDay)]

def getClockCurrentDay(currenttime):
    #currenttime is in seconds
    currentHour = currenttime / sInAnHour
    return currentHour % 24

def PlaneGen(env, runways):
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
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        delayed = random.uniform(0.0,1.0)
        delay = random.gammavariate(3.0, X_delay_expected) if delayed > P_delay and X_delay_expected > 0 else 0
        Plane(env, runways, delay) # should scedule with delay
        planeArrivalTime = max(numpy.random.exponential(getCurrentArrivalIntensity(env.now)),T_guard) + delay
        interarrival_times.append(round(planeArrivalTime, 1))
        arrival_times.append(env.now)
        yield env.timeout(int(planeArrivalTime))


class Plane(object):
    
    info = []
    nr = 0
    
    def __init__(self, env, runways, delay):
        self.env = env
        self.action = env.process(self.run(delay))
        self.runways = runways
        Plane.nr += 1
        self.number = Plane.nr

    
    def add_info(self, more_info):
        self.info.append(more_info)

    def run(self, delay):
        if delay > 0:
            yield delay
        #request runway priority 1
        request_landing = self.runways.request(priority=1)
        yield request_landing
        #yield landing
        yield self.env.timeout(T_landing)
        #release runway
        self.runways.release(request_landing)
        #landed
        landed = self.env.now
        #yield turnaround 
        turnaround = random.gammavariate(7.0, X_turnaround_expected)
        yield self.env.timeout(turnaround)
        #request runway priority 2
        request_takeoff = self.runways.request(priority=2)
        yield request_takeoff
        #yield takeoff-time
        yield self.env.timeout(T_takeoff)
        #release runway
        self.runways.release(request_takeoff)
        # left
        left = self.env.now
        # report variables
        self.add_info([self.number, landed, left])
        

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

def print_bar_diagram(means, standard_deviations, ylabel):
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
    ax.set_ylabel(ylabel)
    ax.set_xticks(length)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

def run_simulation():
    # Code has been refactored to remove the loop. The functions should be able to handle more days without problems
    runways = sim.PriorityResource(env,capacity = N_runways)
    gen = PlaneGen(env,runways)
    env.process(gen)
    env.run(until=SIM_TIME)
    """ results = [interarrival_times, arrival_times]
    mean, stddev = calculate_intervals(results)
    print_bar_diagram(mean, stddev, 'Time between arrival and landing [s]') """
    print(Plane.info)
    return 0

#print(interarrival_times)

run_simulation()
# print_regular_plot(60, max(interarrival_times)+50, interarrival_times, "Interarrival Time [s]", "Hour", numpy.array(arrival_times)/3600, SIM_TIME/3600)
"""
Spørsmål:


"""
