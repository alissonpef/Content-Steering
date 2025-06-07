import docker
import threading


class ContainerMonitor:
    def __init__(self):
        self.client = docker.from_env()
        self.container_stats = {}
        self.nodes = []
        self.interval = 2
        self.choice_algorithm = None
        self.latency_average = []

    def start_collecting(self):
        self.collect_stats()
        threading.Timer(self.interval, self.start_collecting).start()

    def collect_stats(self):
        for container in self.client.containers.list(all=True):
            if container.status != "running":
                if container.name in self.container_stats:
                    del self.container_stats[container.name]
                continue

            try:
                stats = container.stats(stream=False)
                networks = container.attrs["NetworkSettings"]["Networks"]
                ip_address = networks.get("video-streaming_default", {}).get(
                    "IPAddress", "N/A"
                )

                latitude, longitude = None, None
                for env_var in container.attrs["Config"]["Env"]:
                    if env_var.startswith("LATITUDE="):
                        latitude = float(env_var.split("=", 1)[1])
                    elif env_var.startswith("LONGITUDE="):
                        longitude = float(env_var.split("=", 1)[1])

                prev_stats = self.container_stats.get(container.name, [{}])[-1]

                container_stats = {
                    "cpu_usage": stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    / stats["cpu_stats"]["system_cpu_usage"]
                    * 100,
                    "mem_usage": stats["memory_stats"]["usage"]
                    / stats["memory_stats"]["limit"]
                    * 100,
                    "rx_bytes": stats["networks"]["eth0"]["rx_bytes"],
                    "tx_bytes": stats["networks"]["eth0"]["tx_bytes"],
                    "rate_rx_bytes": (
                        stats["networks"]["eth0"]["rx_bytes"]
                        - prev_stats.get("rx_bytes", 0)
                    ),
                    "rate_tx_bytes": (
                        stats["networks"]["eth0"]["tx_bytes"]
                        - prev_stats.get("tx_bytes", 0)
                    ),
                    "ip_address": ip_address,
                    "latitude": latitude,
                    "longitude": longitude,
                }

                if container.name not in self.container_stats:
                    self.container_stats[container.name] = []

                self.container_stats[container.name].append(container_stats)

                self.container_stats[container.name] = self.container_stats[
                    container.name
                ][-10:]

            except Exception as e:
                print(f"Failed to get stats for container {container.name}: {str(e)}")


    def getNodes(self):
        return [
            (name, stat[-1]["ip_address"])
            for name, stat in self.container_stats.items()
        ]

    def get_container_data(self, container_name, data_key):
        if container_name in self.container_stats:
            latest_stats = self.container_stats[container_name][-1]
            return latest_stats.get(data_key)
        return None

    def print_stats(self):
        for name, stats_list in self.container_stats.items():
            print(f"Stats for {name}:")
            if stats_list:
                stats = stats_list[-1]
                print(f"  CPU Usage: {stats['cpu_usage']}")
                print(f"  Memory Usage: {stats['mem_usage']}")
                print(f"  Network Input: {stats['rx_bytes']}")
                print(f"  Network Output: {stats['tx_bytes']}")
                print(f"  Rate Network Input: {stats['rate_rx_bytes']}")
                print(f"  Rate Network Output: {stats['rate_tx_bytes']}")
                print(f"  Metrics size: {len(stats_list)}")
                print(f"  IP address: {stats['ip_address']}")


if __name__ == "__main__":
    main = ContainerMonitor()
    main.start_collecting()