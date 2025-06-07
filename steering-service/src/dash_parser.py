class DashParser:
    def __init__(self):
        pass

    def build(self, target, nodes, uri, request):
        message = {}
        message["VERSION"] = 1
        message["TTL"] = 5 
        message["RELOAD-URI"] = f"{uri}{request.path}"
        
        pathway_priority_nodes = []
        if nodes: 
            pathway_priority_nodes = [f"{node[0]}" for node in nodes]
        
        message["PATHWAY-PRIORITY"] = pathway_priority_nodes + ["cloud"]

        if nodes:
            message["PATHWAY-CLONES"] = self.pathway_clones(nodes)
        return message

    def pathway_clones(self, nodes):
        clones = []
        for node in nodes:
            clone = {
                "BASE-ID": "cloud", # Assume 'cloud' como base, pode precisar ser dinâmico se o MPD tiver outros BaseURLs
                "ID": f"{node[0]}",
                "URI-REPLACEMENT": {"HOST": f"https://{node[0]}"},
            }
            clones.append(clone)
        return clones