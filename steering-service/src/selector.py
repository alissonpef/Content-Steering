import json
import random

class Selector:
    def __init__(self, monitor=None):
        self.monitor = monitor

class EpsilonGreedy(Selector):
    def __init__(self, epsilon, counts, values, monitor=None):
        super().__init__(monitor=monitor)
        self.epsilon = epsilon
        self.counts = counts if counts is not None else {}
        self.values = values if values is not None else {} 
        self.nodes = [] 

    def initialize(self, arms_names: list):
        if not arms_names:
            self.nodes = []
            self.counts = {}
            self.values = {}
            return

        for arm in arms_names:
            if arm not in self.counts:
                self.counts[arm] = 0
                self.values[arm] = 0.0
        
        current_known_arms = list(self.counts.keys())
        for arm in current_known_arms:
            if arm not in arms_names:
                del self.counts[arm]
                del self.values[arm]
            
        self.nodes = list(self.counts.keys())

    def select_arm(self):
        current_monitor_node_names = [name for name, _ in self.monitor.getNodes() if name]
        if not current_monitor_node_names and not self.nodes:
             return []
        if set(current_monitor_node_names) != set(self.nodes):
            self.initialize(current_monitor_node_names)
        
        if not self.nodes:
            return []

        unvisited_arms = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited_arms:
            random.shuffle(unvisited_arms)
            chosen_unvisited_for_exploration = unvisited_arms[0]
            
            other_nodes = [n for n in self.nodes if n != chosen_unvisited_for_exploration]
            
            if not other_nodes:
                return [chosen_unvisited_for_exploration]

            if random.random() > self.epsilon: 
                sorted_remaining_exploit = sorted(other_nodes, key=lambda node: self.values.get(node, 0.0))
                return [chosen_unvisited_for_exploration] + sorted_remaining_exploit
            else: 
                shuffled_remaining = random.sample(other_nodes, len(other_nodes))
                return [chosen_unvisited_for_exploration] + shuffled_remaining

        if random.random() > self.epsilon: 
            sorted_nodes = sorted(list(self.nodes), key=lambda node: self.values.get(node, float('inf')))
            return sorted_nodes
        else: 
            shuffled_nodes = random.sample(list(self.nodes), len(self.nodes))
            return shuffled_nodes

    def update(self, chosen_arm_name, punishment):
        if chosen_arm_name not in self.counts:
            current_monitor_node_names = [name for name, _ in self.monitor.getNodes() if name]
            if chosen_arm_name in current_monitor_node_names:
                self.initialize(current_monitor_node_names)
                if chosen_arm_name not in self.counts:
                    return
            else:
                return

        self.counts[chosen_arm_name] += 1
        n = self.counts[chosen_arm_name]
        value = self.values[chosen_arm_name]
        
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * float(punishment)
        self.values[chosen_arm_name] = new_value
