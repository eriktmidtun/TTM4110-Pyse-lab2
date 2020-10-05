import simpy as sim
import numpy
import random
import datetime
import statistics
from matplotlib import pyplot as plt
import time

env = sim.Environment()

# Table 1
# Arrival intensity
# time                  = [0,1,2,3,4,5  ,6  ,7  ,8 ,9 ,10,11 ,12 ,13 ,14 ,15,16,17,18,19,20 ,21 ,22 ,23]
arrivalIntensityIndexed = [0, 0, 0, 0, 0, 120, 120, 120, 30, 30,
                           30, 150, 150, 150, 150, 30, 30, 30, 30, 30, 120, 120, 120, 120]
DAYS = 30
SIM_TIME = 60*60*24*DAYS
sInAnHour = 60*60

T_guard = 60  # seconds
P_delay = 0.75  # probability
T_landing = 60  # seconds
T_takeoff = 60  # seconds
X_turnaround_expected = 45*60  # seconds
N_runways = 1  # max runways

plane_number = 0
interarrival_times = []
arrival_times = []

def getCurrentArrivalIntensity(currenttime):
    clockCurrentDay = getClockCurrentDay(currenttime)
    return arrivalIntensityIndexed[int(clockCurrentDay)]

def getClockCurrentDay(currenttime):
    # currenttime is in seconds
    currentHour = currenttime / sInAnHour
    return currentHour % 24

def PlaneGen(env, X_delay_expected, runways):
    while True:
        clock = getClockCurrentDay(env.now)
        if clock < 5:
            yield env.timeout(5*sInAnHour-clock*sInAnHour +1)  # wait til 05:00
            # fix points on the start of day
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        delayed = random.random()
        delay = random.gammavariate(
            3.0, X_delay_expected/3) if delayed <= P_delay and X_delay_expected > 0 else 0
        planeArrivalTime = max(numpy.random.exponential(
            getCurrentArrivalIntensity(env.now)), T_guard)
        interarrival_times.append(round(planeArrivalTime+delay, 1))
        arrival_times.append(env.now)
        Plane(env, runways, delay)  # should scedule with delay
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
        # Timestamp and hold delay
        scheduled_arrival = self.env.now
        if delay > 0:
            yield self.env.timeout(delay)
        
        arrival_finished = self.env.now

        # Initiate landing-sequence and timestamp at end
        request_landing = self.runways.request(priority=1)
        yield request_landing
        yield self.env.timeout(T_landing)
        self.runways.release(request_landing)
        landing_finished = self.env.now

        # Initiate turn_around-sequence and timestamp at end
        turnaround = random.gammavariate(7.0, X_turnaround_expected/7)
        # print("\nturnaround:",turnaround/60,"\n")
        yield self.env.timeout(turnaround)
        turn_around_finished = self.env.now

        # Initiate take_off-sequence and timestamp at end
        request_takeoff = self.runways.request(priority=2)
        yield request_takeoff
        yield self.env.timeout(T_takeoff)
        self.runways.release(request_takeoff)
        take_off_finished = self.env.now

        # Report variables
        self.add_info({"number": self.number,
                       "schedule": scheduled_arrival,
                       "arrival": arrival_finished,
                       "delay": arrival_finished - scheduled_arrival,
                       "landing_finished": landing_finished,
                       "landing_time": landing_finished - arrival_finished,
                       "turn_around_finished": turn_around_finished,
                       "turn_around_time": turn_around_finished - landing_finished,
                       "take_off_finished": take_off_finished,
                       "take_off_time": take_off_finished - turn_around_finished,
                       "airport_time": take_off_finished - arrival_finished
                       })
        
        

def calculate_statistics(results, minTime, maxTime, xkey, ykey):
    # Iterates over the results from the simulation in order to find the proper population to examine
    population = []
    population.clear()
    # print("Min:", minTime, "Max:", maxTime)
    for dictionary in results:
        # Making the data go in the correct bin regardless of which day it is
        if getClockCurrentDay(dictionary[xkey])*3600 >= minTime and getClockCurrentDay(dictionary[xkey])*3600 < maxTime:          
            population.append(dictionary[ykey])
    # Returns the mean and population standard deviation for the population
    if len(population) == 0:
        return 0,0
    return statistics.mean(population), statistics.pstdev(population)

def calculate_intervals(results, xkey, ykey):
    number_of_bins = 24
    mean = [0]*5
    stddev = [0]*5
    for i in range(5, number_of_bins):
        # Find the mean and standard deviation for the current bin and append to the corresponding arrays
        i_mean, i_std = calculate_statistics(
            results, 3600*i, 3600*(i+1), xkey, ykey)
        mean.append(i_mean)
        stddev.append(i_std)
    return mean, stddev

def run_simulation():
    start_time = time.time()
    means_landing = []
    stds_landing = []
    means_takeoff = []
    stds_takeoff = []
    means_airport = []
    stds_airport = []
    delays = [0, 300, 600, 1800 ]
    labels = [
        "μ_delay = {delay} s".format(delay=delays[0]),
        "μ_delay = {delay} s".format(delay=delays[1]),
        "μ_delay = {delay} s".format(delay=delays[2]),
        "μ_delay = {delay} s".format(delay=delays[3])]

    for i in range(4):
        # Creating the enviroment/ Resetting the enviroment for multiple runs
        start_time_one_sim = time.time()
        env = sim.Environment()
        # Create resources
        runways = sim.PriorityResource(env, capacity=N_runways)

        # Create entities
        gen = PlaneGen(env, delays[i], runways)
        env.process(gen)
        print("starting run", i, "\n")

        # Run the simulation until the given time
        env.run(until=SIM_TIME)
        # The results we want to examine er in Plane.info and we clear the array to make sure that it doesn't mess up the next iteration
        results = Plane.info.copy()
        Plane.info.clear()
        Plane.nr = 0
        # Calculates the needed statistics in order to print barchart of landing
        xkey = "arrival"
        ykey = "landing_time"
        mean_landing, stddev_landing = calculate_intervals(results, xkey, ykey)
        means_landing.append(mean_landing)
        stds_landing.append(stddev_landing)
        # Calculates the needed statistics in order to print barchart of takeoff
        ykey = "take_off_time"
        mean_takeoff, stddev_takeoff = calculate_intervals(results, xkey, ykey)
        means_takeoff.append(mean_takeoff)
        stds_takeoff.append(stddev_takeoff)
        # Calculates the needed statistics in order to print barchart of airport
        ykey = "airport_time"
        mean_airport, stddev_airport = calculate_intervals(results, xkey, ykey)
        means_airport.append(getClockCurrentDay(numpy.array(mean_airport))*60)
        stds_airport.append(getClockCurrentDay(numpy.array(stddev_airport))*60)
        results.clear()
        print("---run %i took %s seconds ---" % (i, time.time() - start_time_one_sim))
        
    print("--- %s seconds ---" % (time.time() - start_time))
    multiplot_bar(means_airport, stds_airport, labels, "Arrival time", "Time between arrival and takeoff [minutes]", "Time between arrival and takeoff with P(delay) = {P_delay} , {rw} runways".format(P_delay = P_delay, rw = N_runways), "arrival-takeoff-{delayP}-{rw}R-{d}-new2".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    multiplot_bar(means_landing, stds_landing, labels, "Arrival time", "Time between arrival and landing [s]", "Time between arrival and landing with P(delay) = {P_delay} , {rw} runways".format(P_delay = P_delay, rw = N_runways), "arrival-landing-{delayP}-{rw}R-{d}-new2".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    multiplot_bar(means_takeoff, stds_takeoff, labels, "Arrival time", "Time between turn around and takeoff [s]", "Time between turn around and takeoff with P(delay) = {P_delay} , {rw} runways".format(P_delay = P_delay, rw = N_runways), "TA-takeoff-{delayP}-{rw}R-{d}-new2".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    return 0

def multiplot_bar(means, stds, labels, x_label, y_label, title, filename):
    length = numpy.arange(24)
    colors = ["b", "g", "r", "c", "m", "y"]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    for row in axs:
        for col in row:
            col.set_xticks(length)

    for i in range(2):
        for j in range(2):
            axs[i, j].bar(length, means[2*i+j], yerr=stds[2*i+j],
                          align='center', alpha=0.65, capsize=10, color=colors[2*i+j])
            axs[i, j].set_title(labels[2*i+j])

    for ax in axs.flat:
        ax.set(ylabel=y_label)

    plt.suptitle(title, fontsize=25)
    plt.savefig("lab2/plots/" + filename + ".png", dpi=500)


run_simulation()
