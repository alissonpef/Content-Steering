from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin

from dash_parser import DashParser
from monitor import ContainerMonitor


# DEFINES
STEERING_ADDR = 'steering-service'
STEERING_PORT = 30500
BASE_URI      = f'https://{STEERING_ADDR}:{STEERING_PORT}'


# Create instances of the parsers and the container monitor
dash_parser  = DashParser()
monitor = ContainerMonitor()


class Main:
    def __init__(self):
        """
        """
        self.app = Flask(__name__)
        CORS(self.app)

        @self.app.route('/<name>', methods=['GET'])
        @cross_origin()
        def do_remote_steering(name):
            tar = request.args.get('_DASH_pathway', default='', type=str)
            thr = request.args.get('_DASH_throughput', default=0.0, type=float)

            nodes = monitor.getNodes('ip_address')

            print(nodes)

            data = dash_parser.build(
                target  = tar,
                nodes   = nodes,
                uri     = BASE_URI,
                request = request
            )
            
            print(data)
            return jsonify(data), 200


    def run(self):
        ssl_context = ('steering-service/certs/steering-service.pem', 'steering-service/certs/steering-service-key.pem')
        self.app.run(host=STEERING_ADDR, port=STEERING_PORT, debug=True, ssl_context=ssl_context)

# END CLASS.


# MAIN
if __name__ == '__main__':

    monitor.start_collecting()

    main = Main()
    main.run()
# EOF
