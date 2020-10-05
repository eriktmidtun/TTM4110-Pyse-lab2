import simpy as sim
import numpy
import random
import datetime
import statistics
from matplotlib import pyplot as plt
import time

env = sim.Environment()
# Arrival intensity
# time                  = [0,1,2,3,4,5  ,6  ,7  ,8 ,9 ,10,11 ,12 ,13 ,14 ,15,16,17,18,19,20 ,21 ,22 ,23]
arrivalIntensityIndexed = numpy.array([0, 0, 0, 0, 0, 120, 120, 120, 30, 30,
                           30, 15
                            , 150, 150, 150, 30, 30, 30, 30, 30, 120, 120, 120, 120])

DAYS = 30
SIM_TIME = 60*60*24*DAYS
sInAnHour = 60*60

T_guard = 60  # seconds
P_delay = 0.5  # probability
T_landing = 60  # seconds
T_takeoff = 60  # seconds
#T_plowing = 60  # Time needed to plow a runway in seconds
T_deicing = 10*60 # seconds
X_turnaround_expected = 45*60  # seconds
N_runways = 20  # max runways
#N_deicing_trucks = 1 # max deicing trucks
    
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

def PlaneGen(env, X_delay_expected, runways, deicing_trucks):
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
        Plane(env, runways, deicing_trucks, delay)  # should scedule with delay
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
                       "schedule": scheduled_arrival,
                       "arrival": arrival_finished,
                       "delay": arrival_finished - scheduled_arrival,
                       "landing_finished": landing_finished,
                       "landing_time": landing_finished - arrival_finished,
                       "turn_around_finished": turn_around_finished,
                       "turn_around_time": turn_around_finished - landing_finished,
                       "deicing_finished": deicing_finished,
                       "deicing_time": deicing_finished-turn_around_finished,
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
    mean = []
    stddev = []
    for i in range(0, number_of_bins):
        # Find the mean and standard deviation for the current bin and append to the corresponding arrays
        i_mean, i_std = calculate_statistics(
            results, 3600*i, 3600*(i+1), xkey, ykey)
        mean.append(i_mean)
        stddev.append(i_std)
    return mean, stddev

def run_simulation():
    start_time = time.time()
    # Create the arrays to store information about the landing
    means_landing = []
    stds_landing = []

    # Create the arrays to store information about the deicing
    means_deicing = []
    stds_deicing = []
    
    # Create the arrays to store information about the takeoff
    means_takeoff = []
    stds_takeoff = []
    # Create the arrays to store information about the total time at airport
    means_airport = []
    stds_airport = []

    delays = [0, 300, 600, 1800 ]
    labels = [
        "μ_delay = {delay} s".format(delay=delays[0]),
        "μ_delay = {delay} s".format(delay=delays[1]),
        "μ_delay = {delay} s".format(delay=delays[2]),
        "μ_delay = {delay} s".format(delay=delays[3])]

    N_deicing_trucks = [1, 3, 7, 10]
    labels = [
        "N_deicing_trucks = {N_deicing_truck}".format(N_deicing_truck=N_deicing_trucks[0]),
        "N_deicing_trucks = {N_deicing_truck}".format(N_deicing_truck=N_deicing_trucks[1]),
        "N_deicing_trucks = {N_deicing_truck}".format(N_deicing_truck=N_deicing_trucks[2]),
        "N_deicing_trucks = {N_deicing_truck}".format(N_deicing_truck=N_deicing_trucks[3])]

    T_plowings = [1*60, 3*60, 7*60, 10*60]
    labels = [
        "T_plowing = {T_plowing} s".format(T_plowing=T_plowings[0]),
        "T_plowing = {T_plowing} s".format(T_plowing=T_plowings[1]),
        "T_plowing = {T_plowing} s".format(T_plowing=T_plowings[2]),
        "T_plowing = {T_plowing} s".format(T_plowing=T_plowings[3])]

    for i in range(4):
        # Creating the enviroment/ Resetting the enviroment for multiple runs
        start_time_one_sim = time.time()
        env = sim.Environment()

        # Create resources
        runways = sim.PriorityResource(env, capacity=N_runways)
        deicing_trucks = sim.Resource(env, capacity=100)
        snow_container = sim.Container(env, capacity = 1)

        # Create entities
        gen = PlaneGen(env, 0, runways, deicing_trucks)
        plowtruck = PlowTruck(env,snow_container,runways,T_plowings[i] )
        weather = Weather(env,snow_container)
        env.process(gen)
        env.process(plowtruck)
        env.process(weather)

        print("starting run", i, "\n")

        # Run the simulation until the given time
        env.run(until=SIM_TIME)
        # The results we want to examine er in Plane.info and we clear the array to make sure that it doesn't mess up the next iteration
        results = Plane.info.copy()
        Plane.info.clear()
        Plane.nr = 0
        # Calculates the needed statistics in order to print barchart of landing
        xkey = "schedule"
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

        # Calculates the needed statistics in order to print barchart of airport
        ykey = "airport_time"
        mean_airport, stddev_airport = calculate_intervals(results, xkey, ykey)
        means_airport.append(getClockCurrentDay(numpy.array(mean_airport))*60)
        stds_airport.append(getClockCurrentDay(numpy.array(stddev_airport))*60)
        results.clear()
        print("---run %i took %s seconds ---" % (i, time.time() - start_time_one_sim))
    print("--- %s seconds ---" % (time.time() - start_time))
    multiplot_bar(means_airport, stds_airport, labels, "Arrival time", "Time between arrival and takeoff [minutes]", "Time between arrival and takeoff with {rw} runways".format(rw = N_runways), "arrival-takeoff-{delayP}-{rw}R-{d}-snow".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    """ multiplot_bar(means_landing, stds_landing, labels, "Arrival time",
                  "Time between arrival and landing [s]", "Time between arrival and landing with {rw} runways".format(rw = N_runways), "arrival-landing-{delayP}-{rw}R-{d}-snow".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    multiplot_bar(means_deicing, stds_deicing, labels, "Arrival time",
                  "Time between turn around and deicing [s]", "Time between turn around and deicing with {rw} runways".format(rw = N_runways), "TA-deicing-{delayP}-{rw}R-{d}-snow".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS))
    multiplot_bar(means_takeoff, stds_takeoff, labels, "Arrival time",
                  "Time between deicing and takeoff [s]", "Time between deicing and takeoff with {rw} runways".format( rw = N_runways), "deicing-takeoff-{delayP}-{rw}R-{d}-snow".format(delayP = round(100*P_delay), rw = N_runways, d = DAYS)) """
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
    f_name = "lab2/plots/" + filename + ".png"
    print("saved plot as ", f_name)
    plt.savefig(f_name, dpi=500)

def Weather(env, snow_container):
    # Expected values for the random variables
    time_clear_expected = 2*60*60  # Expected length of clear weather in seconds
    time_snowing_expected = 60*60  # Expected length of snow
    snow_intensity_expected = 45*60
    # While the simulation is active...
    while True:
        # Generate the random variables
        time_clear = numpy.random.exponential(1)*time_clear_expected
        time_snowing = numpy.random.exponential(1)*time_snowing_expected
        snow_intensity = numpy.random.exponential(1)*snow_intensity_expected # Tiden været trenger på å fylle rullebanene med snø i sekunder
        # Hold until it starts snowing
        print("Time_clear:", time_clear)
        print("Time_snowing:", time_snowing)
        print("Snow_intensity:", snow_intensity) 
        print("\n\n")
        yield env.timeout(time_clear)
        # Iterate over the time spent snowing to somewhat continously increase the amount of snow
        while(time_snowing > 2.5*60):
            yield env.timeout(5*60)
            #print("test")
            if snow_container.level < snow_container.capacity:
                sim.resources.container.ContainerPut(
                snow_container, min(5*60/snow_intensity, 1-snow_container.level))
            time_snowing -= 5*60

def PlowTruck(env, snow_container, runways, plowing_time):
    # Parameters given by assignment
    # While the simulation is active...
    plowed = 0
    while True:
        # Wait until the runways are covered in snow
        yield sim.resources.container.ContainerGet(snow_container, 1)
        #print("Preparing for plowing. Time =", getClockCurrentDay(env.now))
        # Create an array of runways and request all runways with highest priority
        runway_array = []
        for i in range(N_runways):
            runway_array.append(runways.request(priority=0))
        # Wait for request, plow when available, release when plowed
        start = env.now
        for i in range(N_runways):
            #print("plowing runway:", i)
            yield runway_array[i]
            yield env.timeout(plowing_time)
            runways.release(runway_array[i])
            #print("Runways in use:", runways.count)
        plowed += 1
        print(plowed,"it took",env.now-start, "s to clear all runways")

run_simulation()


#The second factor is the number of plow trucks compared to the number of runways. E.g. If we have so many runways that the plow truck is unable to regularly clear them all before they’re filled again, the extra runways are pretty much useless. In addition, having more plow trucks than runways is just as wasteful. As we’re only able to use a single plow truck per runway in this model, there’s no reason to have more plow trucks than runways. One would therefore want a number of plow trucks that cleanly divide the number of runways. This is due to the fact that all plow trucks would either be working or in standby at the same time. Increasing the amount of plow trucks is the only thing airport management can realistically do to handle snow and as such, we would recommend running a single truck if the time required for plowing is below three minutes as the delays in the top graphs are fine, but start 