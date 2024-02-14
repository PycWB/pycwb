import os
import http.server
import socketserver
import socket
import webbrowser


def find_unoccupied_port(starting_port):
    port = starting_port
    while True:
        if not is_port_in_use(port):
            return port
        port += 1


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_server(port, directory):
    os.chdir(directory)  # Change working directory to your folder

    # Create an HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    print(f"Open http://localhost:{port}/viewer.html in your browser")
    print("------------------------")
    print("| Press Ctrl+C to quit |")
    print("------------------------")
    webbrowser.open(f'http://localhost:{port}/viewer.html')
    httpd.serve_forever()



def init_parser(parser):
    parser.add_argument('dir', type=str, help='Directory to serve')


def command(args):
    port = find_unoccupied_port(8080)
    start_server(port, args.dir)