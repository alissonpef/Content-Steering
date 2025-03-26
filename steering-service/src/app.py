import randomname

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin

from dash_parser import DashParser
from monitor import ContainerMonitor
from selector import Selector, EpsilonGreedy
from environment import Environment

# DEFINES
STEERING_ADDR = 'steering-service'
STEERING_PORT = 30500
BASE_URI      = f'https://0.0.0.0:{STEERING_PORT}'


# Create instances of the parsers and the container monitor
dash_parser  = DashParser()
monitor = ContainerMonitor()
# selector = Selector(monitor)
selector = EpsilonGreedy(0.3, None, None, monitor)

env = Environment()


class Main:
    def __init__(self):
        """
        """
        self.app = Flask(__name__)
        CORS(self.app)

        @self.app.route('/<name>', methods=['GET', 'POST'])
        @cross_origin()
        def do_remote_steering(name):
            tar = request.args.get('_DASH_pathway', default='', type=str)
            thr = request.args.get('_DASH_throughput', default=0.0, type=float)
            uid = request.args.get('_DASH_uid', default=randomname.get_name(), type=str)
            vid = request.args.get('_DASH_video', default='', type=str)

            kwargs = {
                'uid': uid,
                'vid': vid,
                'adr': request.remote_addr,
            }

            nodes = selector.solve(
                monitor.getNodes(), 
                **kwargs
            )

            data = dash_parser.build(
                target  = tar,
                nodes   = nodes,
                uri     = BASE_URI,
                request = request
            )
            
            print(data)
            return jsonify(data), 200
        
        @self.app.route('/coords', methods=['POST'])
        def coords():
            if not request.json:
                return "Invalid request", 400
            
            if not selector.start_algorithm:
                nodes = monitor.getNodes()
                selector.initialize([
                    name for (name, _) in nodes
                ])
            
            coordinates = request.json
            selector.sort_by_coord(coordinates['lat'], coordinates['long'])
            
            return "success", 200


    def run(self):
        ssl_context = (
            'steering-service/certs/steering-service.pem', 
            'steering-service/certs/steering-service-key.pem'
        )
        self.app.run(
            host=STEERING_ADDR, 
            port=STEERING_PORT, 
            debug=True, 
            ssl_context=ssl_context
        )

# END CLASS.


# MAIN
if __name__ == '__main__':

    monitor.start_collecting()

    main = Main()
    main.run()
# EOF
