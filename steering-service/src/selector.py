import random
import math
import logging

selector_logger = logging.getLogger("SelectorStrategies")
if not selector_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    selector_logger.addHandler(_handler)
    selector_logger.setLevel(logging.WARNING)

class Selector:
    # Define a interface para estratégias de seleção de servidor
    def __init__(self, monitor=None, latency_oracle=None):
        self.monitor = monitor
        self.latency_oracle = latency_oracle
        self.nodes = []

    # Configura os servidores (braços) disponíveis
    def initialize(self, arms_names: list):
        self.nodes = [str(arm) for arm in arms_names if arm is not None] if arms_names else []

    # Escolhe um servidor ou uma lista ordenada deles. Implementado por subclasses
    def select_arm(self) -> list:
        raise NotImplementedError

    # Atualiza o seletor com o feedback do servidor escolhido. Opcional
    def update(self, chosen_arm_name: str, feedback_value: float):
        pass

class EpsilonGreedy(Selector):
    # Seleciona o melhor servidor (menor latência estimada) ou um aleatório (exploração)
    def __init__(self, epsilon: float, counts: dict, values: dict, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.epsilon = epsilon
        self.counts = counts if isinstance(counts, dict) else {}
        self.values = values if isinstance(values, dict) else {}

    # Configura os servidores, mantendo contagens e valores existentes
    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values

    # Ordena servidores, priorizando não visitados, depois explora ou explota
    def select_arm(self) -> list:
        if self.monitor:
            current_monitor_node_names = [name for name, _ in self.monitor.getNodes() if name]
            if not current_monitor_node_names and not self.nodes: return []
            if set(current_monitor_node_names) != set(self.nodes):
                self.initialize(current_monitor_node_names)
        if not self.nodes: return []

        unvisited_arms = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited_arms:
            random.shuffle(unvisited_arms)
            chosen_unvisited = unvisited_arms[0]
            other_nodes = [n for n in self.nodes if n != chosen_unvisited]
            if not other_nodes: return [chosen_unvisited]
            order_key = lambda node: self.values.get(node, float("inf"))
            sorted_remaining = sorted(other_nodes, key=order_key) if random.random() > self.epsilon else random.sample(other_nodes, len(other_nodes))
            return [chosen_unvisited] + sorted_remaining

        order_key = lambda node: self.values.get(node, float("inf"))
        return sorted(list(self.nodes), key=order_key) if random.random() > self.epsilon else random.sample(self.nodes, len(self.nodes))

    # Atualiza a latência média estimada para o servidor escolhido
    def update(self, chosen_arm_name: str, punishment: float):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.counts:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes: self.initialize(nodes)
                if str_arm not in self.counts:
                    selector_logger.warning(f"[EpsilonGreedy] Braço {str_arm} não encontrado para update."); return
            else: selector_logger.warning(f"[EpsilonGreedy] Braço {str_arm} desconhecido."); return

        self.counts[str_arm] = self.counts.get(str_arm, 0) + 1
        n = self.counts[str_arm]
        self.values[str_arm] = ((n - 1) / n) * self.values.get(str_arm, 0.0) + (1 / n) * float(punishment)

class NoSteeringSelector(Selector):
    # Retorna servidores em ordem alfabética. Sem steering ativo
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)

    # Retorna servidores disponíveis, ordenados alfabeticamente
    def select_arm(self) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        return sorted(list(self.nodes)) if self.nodes else []

class RandomSelector(Selector):
    # Retorna servidores em ordem aleatória
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)

    # Retorna servidores disponíveis, em ordem aleatória
    def select_arm(self) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []
        return random.sample(self.nodes, len(self.nodes))

class UCB1Selector(Selector):
    # Usa UCB1 para balancear exploração e explotação baseado em recompensa
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.counts = {}
        self.values = {} 
        self.total_pulls = 0

    # Configura servidores, mantendo contagens, valores e total de pulls
    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        self.counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        self.values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        self.total_pulls = sum(self.counts.values())

    # Seleciona servidores por UCB, priorizando não visitados. Maior UCB é melhor
    def select_arm(self) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes: return []
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []

        for arm_name in self.nodes:
            if self.counts.get(arm_name, 0) == 0:
                other_nodes = [n for n in self.nodes if n != arm_name]
                random.shuffle(other_nodes)
                return [arm_name] + other_nodes

        self.total_pulls += 1
        log_total = math.log(self.total_pulls + 1e-5)
        ucb_values = {}
        for arm in self.nodes:
            count = self.counts.get(arm, 1e-5)
            reward = self.values.get(arm, 0.0)
            ucb_values[arm] = reward + math.sqrt((2 * log_total) / count)
        return sorted(ucb_values, key=ucb_values.get, reverse=True)

    # Atualiza a recompensa média (inverso da latência) para o servidor escolhido
    def update(self, chosen_arm_name: str, latency_ms: float):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.counts:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes: self.initialize(nodes)
                if str_arm not in self.counts:
                    selector_logger.warning(f"[UCB1] Braço {str_arm} não encontrado para update."); return
            else: selector_logger.warning(f"[UCB1] Braço {str_arm} desconhecido."); return

        reward = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        self.counts[str_arm] = self.counts.get(str_arm, 0) + 1
        n = self.counts[str_arm]
        self.values[str_arm] = ((n - 1) / n) * self.values.get(str_arm, 0.0) + (1 / n) * reward

class OracleBestChoiceSelector(Selector):
    # Seleciona o servidor com a menor latência atual, via DynamicLatencyOracle
    def __init__(self, monitor=None, latency_oracle=None):
        if latency_oracle is None: raise ValueError("OracleBestChoiceSelector requer DynamicLatencyOracle.")
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)

    # Retorna servidores ordenados pela menor latência atual do oráculo
    def select_arm(self) -> list:
        if not self.latency_oracle:
            selector_logger.warning("[OracleBest] Oráculo indisponível."); return sorted(list(self.nodes)) if self.nodes else []

        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes: return []
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []

        latencies = self.latency_oracle.get_all_current_latencies()
        node_lats = {node: latencies.get(node, float('inf')) for node in self.nodes if node in latencies}
        if not node_lats: return sorted(list(self.nodes)) if self.nodes else []
        return sorted(node_lats, key=node_lats.get)