# -*- coding:utf-8 -*-

import numpy as np
import struct
from typing import Optional
import socket


class PackAge:

    def __init__(self, data: np.ndarray, im_id: int):
        self.image = data.astype(np.uint8)
        self.shape = data.shape
        self.id = im_id

    def to_bytes(self):
        ret = b"yas_yas_mt"
        ret += struct.pack("<i", self.id)
        ret += struct.pack("<I", len(self.image.shape))
        for i in self.image.shape:
            ret += struct.pack("<I", i)
        ret += self.image.tobytes()
        ret += b"\n"
        return ret

    @staticmethod
    def unpack(b: bytes)->[bool, Optional[np.ndarray], int]:
        if b"yas_yas_mt" == b[0:10]:
            _id = struct.unpack("<i", b[10:14])[0]
            tmp = struct.unpack("<I", b[14:18])[0]
            sp = [struct.unpack("<I", b[18+4*i:22+4*i])[0] for i in range(tmp)]
            size = np.prod(sp)*1
            if (b[18+4*tmp+size] == ord("\n")) & (size > 0):
                array = np.fromstring(b[18+4*tmp: 18+4*tmp+size], dtype=np.uint8)
                array = array.reshape(sp)
                return True, array, _id
        return False, None, 0


class UdpClient:
    def __init__(self, server_ip: str, server_port: int, mid: int):
        self.target = (server_ip, server_port)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.id = mid

    def send_udp(self, array: np.ndarray):
        pk = PackAge(array, self.id)
        data = pk.to_bytes()
        self.sk.sendto(data, self.target)
