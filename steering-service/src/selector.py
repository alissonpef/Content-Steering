import json
import random
import math


class Selector:
    def __init__(self, monitor=None):
        self.monitor = monitor
        self.nodes = []

    def initialize(self, arms_names: list):
        if not arms_names:
            self.nodes = []
            return

        self.nodes = [str(arm) for arm in arms_names if arm is not None]

    def select_arm(self):

        raise NotImplementedError


class EpsilonGreedy(Selector):
    def __init__(self, epsilon, counts, values, monitor=None):
        super().__init__(monitor=monitor)
        self.epsilon = epsilon

        self.counts = counts if isinstance(counts, dict) else {}
        self.values = values if isinstance(values, dict) else {}

    def initialize(self, arms_names: list):
        super().initialize(arms_names)

        new_counts = {}
        new_values = {}
        for arm in self.nodes:
            new_counts[arm] = self.counts.get(arm, 0)
            new_values[arm] = self.values.get(arm, 0.0)
        self.counts = new_counts
        self.values = new_values

    def select_arm(self):

        current_monitor_node_names = [
            name for name, _ in self.monitor.getNodes() if name
        ]

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

            other_nodes = [
                n for n in self.nodes if n != chosen_unvisited_for_exploration
            ]

            if not other_nodes:
                return [chosen_unvisited_for_exploration]

            if random.random() > self.epsilon:

                sorted_remaining_exploit = sorted(
                    other_nodes, key=lambda node: self.values.get(node, float("inf"))
                )
                return [chosen_unvisited_for_exploration] + sorted_remaining_exploit
            else:
                shuffled_remaining = random.sample(other_nodes, len(other_nodes))
                return [chosen_unvisited_for_exploration] + shuffled_remaining

        if random.random() > self.epsilon:

            sorted_nodes = sorted(
                list(self.nodes), key=lambda node: self.values.get(node, float("inf"))
            )
            return sorted_nodes
        else:
            shuffled_nodes = random.sample(list(self.nodes), len(self.nodes))
            return shuffled_nodes

    def update(self, chosen_arm_name, punishment):
        str_chosen_arm_name = str(chosen_arm_name)
        if str_chosen_arm_name not in self.counts:
            current_monitor_node_names = [
                name for name, _ in self.monitor.getNodes() if name
            ]
            if str_chosen_arm_name in current_monitor_node_names:
                self.initialize(current_monitor_node_names)
                if str_chosen_arm_name not in self.counts:
                    print(
                        f"[EpsilonGreedy] Warning: Arm {str_chosen_arm_name} not found after re-init for update."
                    )
                    return
            else:
                print(
                    f"[EpsilonGreedy] Warning: Arm {str_chosen_arm_name} for update not in monitor list."
                )
                return

        self.counts[str_chosen_arm_name] = self.counts.get(str_chosen_arm_name, 0) + 1
        n = self.counts[str_chosen_arm_name]
        value = self.values.get(str_chosen_arm_name, 0.0)

        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * float(punishment)
        self.values[str_chosen_arm_name] = new_value


class NoSteeringSelector(Selector):
    def __init__(self, monitor=None):
        super().__init__(monitor=monitor)

    def initialize(self, arms_names: list):
        super().initialize(arms_names)

    def select_arm(self):
        current_monitor_node_names = [
            name for name, _ in self.monitor.getNodes() if name
        ]
        if set(current_monitor_node_names) != set(self.nodes):
            self.initialize(current_monitor_node_names)

        return sorted(list(self.nodes)) if self.nodes else []


class RandomSelector(Selector):
    def __init__(self, monitor=None):
        super().__init__(monitor=monitor)

    def initialize(self, arms_names: list):
        super().initialize(arms_names)

    def select_arm(self):
        current_monitor_node_names = [
            name for name, _ in self.monitor.getNodes() if name
        ]
        if set(current_monitor_node_names) != set(self.nodes):
            self.initialize(current_monitor_node_names)

        if not self.nodes:
            return []

        shuffled_nodes = random.sample(list(self.nodes), len(self.nodes))
        return shuffled_nodes


class UCB1Selector(Selector):
    def __init__(self, monitor=None):
        super().__init__(monitor=monitor)
        self.counts = {}
        self.values = {}
        self.total_pulls = 0

    def initialize(self, arms_names: list):
        super().initialize(arms_names)

        new_counts = {}
        new_values = {}
        for arm in self.nodes:
            new_counts[arm] = self.counts.get(arm, 0)
            new_values[arm] = self.values.get(arm, 0.0)
        self.counts = new_counts
        self.values = new_values

        self.total_pulls = sum(self.counts.values())

    def select_arm(self):
        current_monitor_node_names = [
            name for name, _ in self.monitor.getNodes() if name
        ]
        if not current_monitor_node_names and not self.nodes:
            return []
        if set(current_monitor_node_names) != set(self.nodes):
            self.initialize(current_monitor_node_names)

        if not self.nodes:
            return []

        self.total_pulls += 1

        for arm_name in self.nodes:
            if self.counts.get(arm_name, 0) == 0:
                other_nodes = [n for n in self.nodes if n != arm_name]

                if other_nodes:
                    random.shuffle(other_nodes)
                    return [arm_name] + other_nodes
                else:
                    return [arm_name]

        ucb_values = {}
        log_total_pulls = math.log(self.total_pulls) if self.total_pulls > 0 else 0

        for arm_name in self.nodes:
            count_arm = self.counts.get(arm_name, 0)
            if count_arm > 0:
                average_reward = self.values.get(arm_name, 0.0)

                exploration_term = math.sqrt(
                    (2 * (log_total_pulls + 1e-5)) / (count_arm + 1e-5)
                )
                ucb_values[arm_name] = average_reward + exploration_term
            else:

                ucb_values[arm_name] = float("inf")

        sorted_nodes_by_ucb = sorted(
            ucb_values.keys(), key=lambda arm: ucb_values[arm], reverse=True
        )
        return sorted_nodes_by_ucb

    def update(self, chosen_arm_name, latency_ms):
        str_chosen_arm_name = str(chosen_arm_name)
        if str_chosen_arm_name not in self.counts:
            current_monitor_node_names = [
                name for name, _ in self.monitor.getNodes() if name
            ]
            if str_chosen_arm_name in current_monitor_node_names:
                self.initialize(current_monitor_node_names)
                if str_chosen_arm_name not in self.counts:
                    print(
                        f"[UCB1] Warning: Arm {str_chosen_arm_name} not found after re-init for update."
                    )
                    return
            else:
                print(
                    f"[UCB1] Warning: Arm {str_chosen_arm_name} for update not in monitor list."
                )
                return

        reward = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        self.counts[str_chosen_arm_name] = self.counts.get(str_chosen_arm_name, 0) + 1
        n = self.counts[str_chosen_arm_name]

        current_average_reward = self.values.get(str_chosen_arm_name, 0.0)
        new_average_reward = ((n - 1) / float(n)) * current_average_reward + (
            1 / float(n)
        ) * reward
        self.values[str_chosen_arm_name] = new_average_reward
