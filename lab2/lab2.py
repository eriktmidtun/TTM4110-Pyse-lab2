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

SIM_TIME = 60*60*24*3 #one day
T_guard = 60 #seconds
P_delay = 0.5 # probability
sInAnHour = 60*60

interarrival_times = []
arrival_times = []

def printplot(ymin, ymax, y, ylabel, xlabel, x, xmax):
    baneform = plt.figure("",figsize=(12,3))
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
            arrival_times.append(env.now / 3600)
            yield env.timeout(5*sInAnHour-clock*sInAnHour) # wait til 05:00
            #fix points on the start of day
            interarrival_times.append(T_guard)
            arrival_times.append(env.now / 3600)
        plane += 1
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        #Plane(env)
        delayed = random.uniform(0.0,1.0)
        delay = random.gammavariate(3.0, X_delay_expected) if delayed > P_delay and X_delay_expected > 0 else 0
        planeArrivalTime = max(numpy.random.exponential(getCurrentArrivalIntensity(env.now)),T_guard) + delay
        interarrival_times.append(round(planeArrivalTime, 1))
        arrival_times.append(env.now / 3600)
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


gen = PlaneGen(env)
env.process(gen)
#start two runaway processes?
env.run(until=SIM_TIME)

def calculate_std_dev(results, minTime, maxTime):
    population = []
    for i in range(len(results)):
        if results[1][i] <= maxTime:
            break
        elif results[1][i] >= minTime:
            population.append(results[0][i])
    return statistics.pstdev(population)

def calculate_mean(results, minTime, maxTime):
    population = []
    for i in range(len(results)):
        if results[1][i] <= maxTime:
            break
        elif results[1][i] >= minTime:
            population.append(results[0][i])
    return statistics.mean(population)

def calculate_intervals(results, length):
    number_of_bins = 24/length
    mean_bins = [[], []]
    stdev_bins = [[], []]
    less_mean_bins = [[], []]
    more_mean_bins = [[], []]
    for i in range(number_of_bins):
        stdev = calculate_std_dev(results, i*number_of_bins, (i+1)*number_of_bins)
        stdev_bins[0].append(stdev)
        stdev_bins[1].append(i*number_of_bins)
        mean = calculate_mean(results, i*number_of_bins, (i+1)*number_of_bins)
        mean_bins[0].append(mean)
        mean_bins[1].append(i*number_of_bins)
    less_mean_bins[0] = numpy.array()
    return 0
    
def run_simulation(number):
    results = [[]*number, []*number]
    for i in range(number):
        interarrival_times = []
        arrival_times = []
        gen = PlaneGen(env)
        env.process(gen)
        env.run(until=SIM_TIME)
        results[0][i] = interarrival_times
        results[1][i] = arrival_times
    return 0
        

""" print(interarrival_times)
plt.plot(numpy.array(interarrival_times))
plt.xlabel("Plane Number")
plt.ylabel("Delay [s]") 
plt.show() """

#print(interarrival_times)
printplot(60, max(interarrival_times)+50, interarrival_times, "Interarrival Time [s]", "Hour", arrival_times, SIM_TIME/3600)
""" print("Mean:", numpy.mean(duration_of_successful_calls))
print("Number of missed calls:",calls["Failures"], "Probability of failure:", 100*calls["Failures"]/calls["Number"],"%") """


"""
Spørsmål:


"""