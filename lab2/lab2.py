import simpy as sim
import numpy
import random
import datetime
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
    X_delay_expected = -10
    while True:
        clock = getClockCurrentDay(env.now)
        if clock < 5:
            X_delay_expected += 10
            interarrival_times.append(60)
            arrival_times.append(env.now / 3600)
            yield env.timeout(5*sInAnHour-clock*sInAnHour) # wait til 05:00
            interarrival_times.append(60)
            arrival_times.append(env.now / 3600)
        plane += 1
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        #Plane(env)
        delayed = random.uniform(0.0,1.0)
        delay = random.gammavariate(3.0, X_delay_expected) if delayed > P_delay and X_delay_expected > 0 else 0 #must check if gammavariate actually is erlang
        planeArrivalTime = max(numpy.random.exponential(getCurrentArrivalIntensity(env.now)),T_guard) + delay
        interarrival_times.append(round(planeArrivalTime, 1))
        arrival_times.append(env.now / 3600)
        yield env.timeout(int(planeArrivalTime))
        
        

gen = PlaneGen(env)
env.process(gen)
env.run(until=SIM_TIME)

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
1. Gradvis øke expected_delay med hvert fly eller øke det ved døgnskifte?
2. Hva ønsker dere at vi skal plotte?
3. Is we big stupid?

"""