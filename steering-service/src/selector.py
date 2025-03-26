import json
import random

import latency.latency_estimator as lat_estimator



NODE_NAME = 0


class Selector:
    def __init__(self, monitor=None):
        self.clouds = []
        self.clusters = []
        self.sessions = {}

        self.start_algorithm = False
        self.monitor = monitor
        
    def solve(self, uid, **kwargs) -> object:
        raise ValueError("Both request problem and select cache problem must be set")



class EpsilonGreedy(Selector):
    def __init__(self, epsilon, counts, values, monitor=None):
        self.epsilon = epsilon
        self.counts  = counts
        self.values  = values
        self.nodes   = None
        self.latency_average = []
        
        super().__init__(monitor=monitor)


    # Initialise arms with given names
    def initialize(self, arms_names : list):
        self.counts = dict.fromkeys(arms_names, 0)
        self.values = dict.fromkeys(arms_names, 0.0)

        self.nodes = arms_names
    
    def select_arm(self):
        if random.random() > self.epsilon:
            return sorted(self.nodes, key=lambda node: self.values[node])

        random.shuffle(self.nodes)
        return self.nodes

    
    def solve(self, nodes, **kwargs):
        return self.select_arm()
    
    
    def update(self, chosen_arm_name, punishment):
        # update counts pulled for chosen arm
        self.counts[chosen_arm_name] = self.counts[chosen_arm_name] + 1
        n = self.counts[chosen_arm_name]
        
        # Update average/mean value/punishment for chosen arm
        value = self.values[chosen_arm_name]
        new_value = ((n-1)/float(n)) * value + (1 / float(n)) * punishment
        self.values[chosen_arm_name] = new_value

        return
    
    
    def sort_by_coord(self, lat, lon):        
        print(f"[LOG][client] lat: {lat}, long: {lon}")

        # Estimate latency
        selected_node      = self.select_arm()[0]
        selected_node_lat  = self.monitor.get_container_data(selected_node, 'latitude')
        selected_node_long = self.monitor.get_container_data(selected_node, 'longitude')
        estimated_latency  = lat_estimator.estimate_latency(lat, lon, selected_node_lat, selected_node_long)
        
        print(
            f"[LOG] Selected Node: {selected_node}, "
            f"lat: {selected_node_lat:.6f}, "
            f"long: {selected_node_long:.6f}, "
            f"latency: {estimated_latency:.2f} ms"
        )

        # Makes update
        self.update(selected_node, estimated_latency)

        self.log_latency_average(estimated_latency)
        
        
    def log_latency_average(self, estimated_latency):
        if not self.latency_average:
            self.latency_average.append(estimated_latency)
        else:
            current_avr = (sum(self.latency_average) + estimated_latency) / (len(self.latency_average) + 1)
            self.latency_average.append(current_avr)
            self.latency_average = self.latency_average[-10:]
        
        with open("logs/latency_average.txt", "a") as arquivo:
            arquivo.write(f"{self.latency_average[-1]}\n")