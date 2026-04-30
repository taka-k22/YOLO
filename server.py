from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        print(self.rfile.read(length).decode())
        self.send_response(200)
        self.end_headers()

HTTPServer(('localhost', 3000), H).serve_forever()