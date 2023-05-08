import socket
import threading
import inspect
import ctypes
import json
import time
from queue import Queue


class Send_data(threading.Thread):
    ip, port = "127.0.0.1", 25001
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    datas = Queue()
    data_is_convey = True

    def __init__(self, con):
        self.con = con
        self.server.bind((self.ip, self.port))
        self.server.listen(5)
        print("[*]成功创建线程!")
        super().__init__()

    def run(self):
        self.con.acquire()
        print("执行socket调用")
        client, addr = self.server.accept()  # 如果有客户端链接
        print("[*]成功建立连接于 %s:%d" % (addr[0], addr[1]))

        received_data = client.recv(1024).decode(
            "UTF-8")  # receiveing data in Byte fron C#, and converting it to String
        print(received_data)

        while not self.datas.empty() or self.data_is_convey:
            self.con.notify()
            self.con.wait()
            if not self.datas.empty():
                data_string = json.dumps(self.datas.get())
                client.send(data_string.encode("UTF-8"))

        client.send(json.dumps({"data_type": "string", "sign": "FINISH"}).encode("UTF-8"))
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        print("通信结束")
        self.con.release()

    def data_save(self, data):
        self.datas.put(data)

    def data_finish(self):
        self.data_is_convey = False
