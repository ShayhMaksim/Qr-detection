import socket

client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

BIND_IP='127.0.0.1'
BIND_PORT=8080


client.connect((BIND_IP,BIND_PORT))


while True:
    response = client.recv(1024)
    print(response)
    