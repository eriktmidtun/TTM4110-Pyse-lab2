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
arrivalIntensityIndexed = [0, 0, 0, 0, 0, 120, 120, 120, 30, 30,
                           30, 150, 150, 150, 150, 30, 30, 30, 30, 30, 120, 120, 120, 120]

SIM_TIME = 60*60*24
sInAnHour = 60*60

T_guard = 60  # seconds
P_delay = 0.5  # probability
T_landing = 60  # seconds
T_takeoff = 60  # seconds
T_deicing = 10*60 # seconds
X_turnaround_expected = 45*60  # seconds
N_runways = 2  # max runways
N_deicing_trucks = 1 # max deicing trucks
N_plow_trucks = 1 # max plow trucks

interarrival_times = []
arrival_times = []


def getCurrentArrivalIntensity(currenttime):
    clockCurrentDay = getClockCurrentDay(currenttime)
    return arrivalIntensityIndexed[int(clockCurrentDay)]

def getClockCurrentDay(currenttime):
    # currenttime is in seconds
    currentHour = currenttime / sInAnHour
    return currentHour % 24

def PlaneGen(env, runways, deicing_trucks):
    X_delay_expected = 0
    while True:
        clock = getClockCurrentDay(env.now)
        if clock < 5:
            #X_delay_expected += 10
            interarrival_times.append(T_guard)  # fix points on the end of day
            arrival_times.append(env.now)
            yield env.timeout(5*sInAnHour-clock*sInAnHour)  # wait til 05:00
            # fix points on the start of day
            interarrival_times.append(T_guard)
            arrival_times.append(env.now)
        #print("Plane %i arrived at %s" % (plane, datetime.timedelta(seconds = env.now)))
        delayed = random.uniform(0.0, 1.0)
        delay = random.gammavariate(
            3.0, X_delay_expected) if delayed > P_delay and X_delay_expected > 0 else 0
        Plane(env, runways, deicing_trucks, delay)  # should scedule with delay
        planeArrivalTime = max(numpy.random.exponential(
            getCurrentArrivalIntensity(env.now)), T_guard) + delay
        interarrival_times.append(round(planeArrivalTime, 1))
        arrival_times.append(env.now)
        yield env.timeout(int(planeArrivalTime))

class Plane(object):

    info = []
    nr = 0

    def __init__(self, env, runways, deicing_trucks, delay):
        self.env = env
        self.action = env.process(self.run(delay))
        self.runways = runways
        self.deicing_trucks = deicing_trucks
        Plane.nr += 1
        self.number = Plane.nr

    def add_info(self, more_info):
        self.info.append(more_info)

    def run(self, delay):
        # Timestamp and hold delay
        arrival_finished = self.env.now
        if delay > 0:
            yield delay
        delay_finished = self.env.now

        # Initiate landing-sequence and timestamp at end
        request_landing = self.runways.request(priority=1)
        yield request_landing
        yield self.env.timeout(T_landing)
        self.runways.release(request_landing)
        landing_finished = self.env.now

        # Initiate turn_around-sequence and timestamp at end
        turnaround = random.gammavariate(7.0, X_turnaround_expected)
        yield self.env.timeout(turnaround)
        turn_around_finished = self.env.now
        
        # Initiate deicing-sequence and timestamp at end
        request_deicing = self.deicing_trucks.request()
        yield request_deicing
        yield self.env.timeout(T_deicing)
        self.deicing_trucks.release(request_deicing)
        deicing_finished = self.env.now

        # Initiate take_off-sequence and timestamp at end
        request_takeoff = self.runways.request(priority=2)
        yield request_takeoff
        yield self.env.timeout(T_takeoff)
        self.runways.release(request_takeoff)
        take_off_finished = self.env.now

        # Report variables
        self.add_info({"number": self.number,
                       "arrival": arrival_finished,
                       "delay_finished": delay_finished,
                       "delay_time": delay_finished - arrival_finished,
                       "landing_finished": landing_finished,
                       "landing_time": landing_finished - arrival_finished,
                       "turn_around_finished": turn_around_finished,
                       "turn_around_time": turn_around_finished - landing_finished,
                       "deicing_finished": deicing_finished,
                       "deicing_time": deicing_finished - turn_around_finished,
                       "take_off_finished": take_off_finished,
                       "take_off_time": take_off_finished - turn_around_finished
                       })

def calculate_statistics(results, minTime, maxTime, xkey, ykey):
    # Iterates over the results from the simulation in order to find the proper population to examine
    population = []
    for dictionary in results:
        # Making the data go in the correct bin regardless of which day it is
        if getClockCurrentDay(dictionary[xkey])*3600 >= minTime and getClockCurrentDay(dictionary[xkey])*3600 < maxTime:
            population.append(dictionary[ykey])
    # Returns the mean and population standard deviation for the population
    if len(population) < 1 or minTime == 0:
        return 0, 0
    return statistics.mean(population), statistics.pstdev(population)

def calculate_intervals(results, xkey, ykey):
    number_of_bins = 24
    mean = []
    stddev = []
    for i in range(number_of_bins):
        # Find the mean and standard deviation for the current bin and append to the corresponding arrays
        i_mean, i_std = calculate_statistics(
            results, 3600*i, 3600*(i+1), xkey, ykey)
        mean.append(i_mean)
        stddev.append(i_std)
    return mean, stddev

def run_simulation():
    # x_values are identical for all examinations
    x_values = []
    # Create the arrays to store information about the landing
    means_landing = []
    stds_landing = []
    # Create the arrays to store information about the deicing
    means_deicing = []
    stds_deicing = []
    # Create the arrays to store information about the takeoff
    means_takeoff = []
    stds_takeoff = []

    # We want to test four different values for delay
    delays = [0, 10, 30, 60]
    labels = [
        "μ_delay = {delay} s".format(delay=delays[0]),
        "μ_delay = {delay} s".format(delay=delays[1]),
        "μ_delay = {delay} s".format(delay=delays[2]),
        "μ_delay = {delay} s".format(delay=delays[3])]

    for i in range(4):
        # Creating the enviroment/ Resetting the enviroment for multiple runs
        env = sim.Environment()
        # Create resources
        runways = sim.PriorityResource(env, capacity=N_runways)
        deicing_trucks = sim.Resource(env, capacity=N_deicing_trucks)

        # Create entities
        gen = PlaneGen(env, runways, deicing_trucks)
        env.process(gen)

        # Run the simulation until the given time
        env.run(until=SIM_TIME)
        # The results we want to examine er in Plane.info and we clear the array to make sure that it doesn't mess up the next iteration
        results = Plane.info.copy()
        Plane.info.clear()
        # Calculates the needed statistics in order to print barchart of landing
        xkey = "arrival"
        ykey = "landing_time"
        mean_landing, stddev_landing = calculate_intervals(results, xkey, ykey)
        means_landing.append(mean_landing)
        stds_landing.append(stddev_landing)
        # Calculates the needed statistics in order to print barchart of deicing
        ykey = "deicing_time"
        mean_deicing, stddev_deicing = calculate_intervals(results, xkey, ykey)
        means_deicing.append(mean_deicing)
        stds_deicing.append(stddev_deicing)
        # Calculates the needed statistics in order to print barchart of takeoff
        ykey = "take_off_time"
        mean_takeoff, stddev_takeoff = calculate_intervals(results, xkey, ykey)
        means_takeoff.append(mean_takeoff)
        stds_takeoff.append(stddev_takeoff)

    multiplot_bar(means_landing, stds_landing, labels, "Arrival time",
                  "Time between arrival and landing [s]", "Time between arrival and landing")
    multiplot_bar(means_landing, stds_landing, labels, "Arrival time",
                  "Time between arrival and landing [s]", "Time between turn around and deicing")
    multiplot_bar(means_takeoff, stds_takeoff, labels, "Arrival time",
                  "Time between turn around and takeoff [s]", "Time between deicing and takeoff")
    return 0

def multiplot_bar(means, stds, labels, x_label, y_label, title):
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
    plt.savefig("lab2/plots/allplots.png", dpi=500)

def Weather(env, snow_container):
    # Expected values for the random variables
    time_clear_expected = 2*60*60  # Expected length of clear weather in seconds
    time_snowing_expected = 60*60  # Expected length of snow
    snow_intensity_expected = 45*60
    # While the simulation is active...
    while True:
        # Generate the random variables
        time_clear = numpy.random.exponential(time_clear_expected)
        time_snowing = numpy.random.exponential(time_snowing_expected)
        snow_intensity = numpy.random.exponential(snow_intensity_expected) # Tiden været trenger på å fylle rullebanene med snø i sekunder
        # Hold until it starts snowing
        yield env.timeout(time_clear)
        # Iterate over the time spent snowing to somewhat continously increase the amount of snow
        while(time_snowing > 2.5*60):
            yield env.timeout(5*60)
            sim.resources.container.ContainerPut(
                snow_container, 5*60/snow_intensity)
            time_snowing -= 5*60

def PlowTruck(env, snow_container, runways):
    # Parameters given by assignment
    time_plowing = 10 * 60  # Time needed to plow a runway in seconds
    # While the simulation is active...
    while True:
        # Wait until the runways are covered in snow
        yield sim.resources.container.ContainerGet(snow_container, 1)
        # Create an array of runways and request all runways with highest priority
        runway_array = []
        for i in range(N_runways):
            runway_array.append(runways.request(priority=0))
        # Wait for request, plow when available, release when plowed
        for i in range(N_runways):
            yield runway_array[i]
            yield env.timeout(time_plowing)
            runways.release(runway_array[i])


run_simulation()
