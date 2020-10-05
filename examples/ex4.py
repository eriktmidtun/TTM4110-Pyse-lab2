import simpy as sim
import numpy

env = sim.Environment()

NEXT_CALL = 30*60 #timeout is in seconds
MAX_CONNECTION_TIME = 15 #30
FIXED_CONNECTION_TIME = 0.200
DISCONNECTION_TIME = 0.200 
AVG_VARIABLE_CONNECTION_TIME = 3.0 #0.2
AVG_CONVERSATION_TIME = 3*60
SIM_TIME = 30*24*60*60
duration_of_successful_calls = []
calls = {"Number": 0, "Failures": 0}

class Subscriber(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            #print("Waiting for next call at %d" % (env.now))
            TIME_BETWEEN_CALLS = numpy.random.exponential(NEXT_CALL)
            yield self.env.timeout(TIME_BETWEEN_CALLS)
            #print("Next call at %d" % (env.now))
            timer = Timer(env, self) #start timer here
            try:
                calls["Number"] += 1
                VARIABLE_CONNECTION_TIME = numpy.random.exponential(AVG_VARIABLE_CONNECTION_TIME) 
                yield self.env.timeout(FIXED_CONNECTION_TIME + VARIABLE_CONNECTION_TIME)
                timer.action.interrupt()
                #print("Connection established at %d" % (env.now))
                CONVERSATION_TIME = numpy.random.exponential(AVG_CONVERSATION_TIME)
                duration_of_successful_calls.append(VARIABLE_CONNECTION_TIME + FIXED_CONNECTION_TIME + CONVERSATION_TIME + DISCONNECTION_TIME)
                yield self.env.timeout(CONVERSATION_TIME)
                #print("Conversation done at %d" % (env.now))
            except sim.Interrupt:
                #print("Connection interrupted")
                calls["Failures"] += 1
            finally:
                #print("Disconnection started at %d" % (env.now))
                yield self.env.timeout(DISCONNECTION_TIME)
                #print("Disconnection finished at %d" % (env.now))
            

class Timer(object):
    def __init__(self, env, sub):
        self.env = env
        self.sub = sub
        self.action = env.process(self.run())

    def run(self):
        #print("Starts timer at %d" % (env.now))
        try:
            yield self.env.timeout(MAX_CONNECTION_TIME)
            #print("Max connection time over at %d" % (env.now))
            self.sub.action.interrupt() # interrupt subscriber connection
        except sim.Interrupt:
            #print("Timer interrupted at %d" % (env.now))
            a = 0

sub = Subscriber(env)
env.run(until=SIM_TIME)
print("Mean:", numpy.mean(duration_of_successful_calls))
print("Number of missed calls:",calls["Failures"], "Probability of failure:", 100*calls["Failures"]/calls["Number"],"%")