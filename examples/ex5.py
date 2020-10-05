import simpy as sim
import numpy
import random

env = sim.Environment()


SIM_TIME = 300
MAX_DELAY = 3
TRANSMISSION_DELAY = 0.2
packet_info = []
calls = {"Number": 0, "Failures": 0}

def PacketGen(env):
    number_of_packets = 0
    while True:
        Packet(env, number_of_packets)
        number_of_packets += 1
        yield self.env.timeout(numpy.random.exponential(2))

class Packet(object):
    def __init__(self, env, packet_nr):
        self.env = env
        self.action = env.process(self.run())
        self.nr = packet_nr

    def run(self):
        packet = {"timestamp": env.now, "ttl": True}
        packet_info.append(packet)
        yield self.env.timeout(TRANSMISSION_DELAY)
        DRAW = random.uniform(0.0,1.0)
        if DRAW > 0.5:
            pick R1
        else:
            Pick R2
        if packet_info[self.nr]["ttl"]: #not discarded
            yield self.env.timeout(TRANSMISSION_DELAY) #transdelay
            wait for r3
        

""" class Router(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            if packet.time > timestamp + MAX_DELAY:
                packet.ttl = False
            else:
                yield self.env.timeout(random.gammavariate(3, 1)) """

        
router1 = Router(env) # maybe use store? or sleep until woken up?
router2 = 
plane = Plane(env)
env.run(until=SIM_TIME)
print("Mean:", numpy.mean(duration_of_successful_calls))
print("Number of missed calls:",calls["Failures"], "Probability of failure:", 100*calls["Failures"]/calls["Number"],"%")