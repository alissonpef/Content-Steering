class DashParser:
    def __init__(self):
        pass


    def build(self, target, nodes, uri, request):
        message = {}
        message['VERSION'] = 1
        message['TTL'] = 5
        message['RELOAD-URI'] = f'{uri}{request.path}'

        message["PATHWAY-PRIORITY"] = [f'{node[0]}' for node in nodes] + ['cloud']

        if nodes:
            message['PATHWAY-CLONES'] = self.pathway_clones(nodes)
        
        return message


    def pathway_clones(self, nodes):
        
        clones = []
        
        for node in nodes:
            clone = {
                'BASE-ID': f'cloud',
                'ID': f'{node[0]}',
                'URI-REPLACEMENT': {
                    'HOST': f'https://{node[0]}'
                }
            }
    
            clones.append(clone)
            
        return clones
