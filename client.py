import base64
import random
import numpy as np
import secrets
import json
import socketio
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sslib import shamir, randomness
import time
import _thread
from model import MLP
from utils import model2array, array2model


def generate_key():
    sk = ec.generate_private_key(ec.SECP384R1(), default_backend())
    pk = sk.public_key()
    return sk, pk


def deriveKey(sk, pk):
    sharedKey = sk.exchange(ec.ECDH(), pk)
    derivedKey = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
        backend=default_backend()
    ).derive(sharedKey)
    key = base64.urlsafe_b64encode(derivedKey)
    return key


class SecAggClient:
    def __init__(self, threadId):
        self.id = ''
        self.threshold = 2
        self.epoch = 0
        self.suSK, self.suPK = generate_key()
        self.cuSK, self.cuPK = generate_key()
        self.bu = secrets.token_bytes(16)
        self.sio = socketio.Client()
        self.clientU1 = []
        self.clientU2 = set()
        self.clientU3 = set()
        self.clientU2Secrets = {}
        self.drop = 0
        self.threadId = threadId
        self.model = None
        self.res = None

    def setDrop(self, d):
        self.drop = d

    def create_handler(self):
        sio = self.sio

        # round 1: upload user suPk and cuPk
        @sio.event()
        def connect():
            # print('connect to server')
            # print(str(self.threadId) + ' send public keys')
            suPK_bytes = self.suPK.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo)
            cuPK_bytes = self.cuPK.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo)
            pubkey = {'epoch': self.epoch,
                      'suPK': str(suPK_bytes, 'utf-8'), 'cuPK': str(cuPK_bytes, 'utf-8')}
            sio.emit('AdvertiseKeys', pubkey)

        @sio.on('connect_success')
        def on_connect_success(data):
            # print('connect success', data)
            self.id = data['id']
            # print('connect to server')
            # print('connect established, my id is ', self.id)

        @sio.on('epoch_error')
        def on_epoch_error():
            sio.disconnect()

        # round 2: compute shared key with other suPk and upload encrypted shares with cuPk
        @sio.on('clientU1')
        def on_clientU1(data):
            if self.drop == 1:
                sio.disconnect()
                return
            self.clientU1 = data
            # print('receive clientU1')
            # print(self.clientU1)
            cipher = self.splitSecrets()
            # print(str(self.threadId) + ' upload secrets')
            sio.emit('ShareKeys', cipher)

        # round 3: upload masked models to server
        @sio.on('clientU2')
        def on_clientU2(data):
            if self.drop == 2:
                sio.disconnect()
                return
            # print(str(self.threadId) + ' get others Secrets from server')
            self.decryptSecrets(data)
            model = self.maskModel()
            sio.emit('postModels', base64.b64encode(model))

        # round 4：upload decrypted bu shares (online user) and suSk share (offline user)
        @sio.on('clientU3')
        def on_clientU3(data):
            # print(str(self.threadId) + ' post Shares')
            self.clientU3 = set(data)
            shares = []
            for sid in self.clientU2:
                if sid in self.clientU3:
                    shares.append({
                        'id': sid,
                        'buShare': self.clientU2Secrets[sid]['buShare'],
                        'suShare': None
                    })
                else:
                    shares.append({
                        'id': sid,
                        'buShare': None,
                        'suShare': self.clientU2Secrets[sid]['suShare']
                    })
            sio.emit('Unmasking', shares)

        @sio.on('finish')
        def on_finish(data):
            r = base64.b64decode(data)
            model = np.frombuffer(r, dtype=np.dtype('float32'))
            # print("WELL DONE! global model:\n{}".format(model))
            self.res = model
            sio.disconnect()

        @sio.event
        def connect_error(x):
            print("The connection failed!")
            sio.disconnect()

        @sio.event
        def disconnect():
            print('disconnected from server')

    def maskModel(self):
        # np.random.seed(self.bu)
        # m = np.random.randn(2, 2)
        # m = np.zeros(10)
        m = self.model
        print(m)
        random.seed(self.bu)
        for i in range(len(m)):
            m[i] += random.random()
        x = 1
        # print('clientU2: ', self.clientU2)
        for c in self.clientU1:
            if c['id'] == self.id:
                x = -1
                continue
            if c['id'] in self.clientU2:
                svPK = serialization.load_pem_public_key(c['suPK'].encode('utf-8'), default_backend())
                shareKey = deriveKey(self.suSK, svPK)
                random.seed(shareKey)
                for i in range(len(m)):
                    m[i] += random.random() * x
        # print(m)
        return m

    def splitSecrets(self):
        clientsU1 = self.clientU1
        shamirBu = shamir.split_secret(
            self.bu,
            self.threshold,
            len(clientsU1) - 1,
            randomness_source=randomness.UrandomReader()
        )
        shamirBu = shamir.to_base64(shamirBu)
        sharesBu = shamirBu['shares']
        suSK_bytes = self.suSK.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        sharmirSk = shamir.split_secret(suSK_bytes, self.threshold, len(clientsU1) - 1)
        sharmirSk = shamir.to_base64(sharmirSk)
        sharesSuSK = sharmirSk['shares']
        i = 0
        ciphertexts = []
        for c in clientsU1:
            if c['id'] != self.id:
                msg = str(self.id) + ';' + str(c['id']) + ';' + sharesSuSK[i] + ';' + sharesBu[i]
                # publicKey = c['cuPK']
                publicKey = serialization.load_pem_public_key(c['cuPK'].encode('utf-8'), default_backend())
                shareKey = deriveKey(self.cuSK, publicKey)
                f = Fernet(shareKey)
                ciphertexts.append({'id': c['id'], 'cipher': str(f.encrypt(msg.encode('utf8')), "utf-8")})
                c['fernet'] = f
                i += 1
        return {
            'suSKPrimeMod': sharmirSk['prime_mod'],
            'buPrimeMod': shamirBu['prime_mod'],
            'ciphertexts': ciphertexts
        }

    def decryptSecrets(self, data):
        # print('\n decrypt Secrets \n')
        for cipher in data:
            tid = cipher['id']
            self.clientU2.add(tid)
            for c in self.clientU1:
                if c['id'] == tid:
                    decryptedCipher = str(c['fernet'].decrypt(cipher['cipher'].encode('utf8')), 'utf8')
                    decryptedCipher = decryptedCipher.split(';')
                    if decryptedCipher[0] == tid and decryptedCipher[1] == self.id:
                        self.clientU2Secrets[tid] = {'suShare': decryptedCipher[2], 'buShare': decryptedCipher[3]}
                    else:
                        print("wrong secrets send to user {}".format(self.id))
                        # self.sio.disconnect()

    def start(self, model, epoch):
        self.model = model
        self.epoch = epoch
        self.sio.connect('http://127.0.0.1:5000')
        self.sio.wait()
        return self.res


def create_client(thread_id, m):
    for i in range(5):
        client = SecAggClient(thread_id)
        client.setDrop(0)
        client.create_handler()
        x, y, z = model2array(m)
        out = client.start(x, i)
        res = array2model(out, y, z)
        print('epoch-{0}:{1}'.format(i, res))


if __name__ == "__main__":
    try:
        mlp = MLP(2, 2, 1).state_dict()
        print(mlp)
        _thread.start_new_thread(create_client, (0, mlp))
        _thread.start_new_thread(create_client, (1, mlp))
        _thread.start_new_thread(create_client, (2, mlp))
        _thread.start_new_thread(create_client, (3, mlp))
    except:
        print("Error: 无法启动线程")

    while 1:
        pass
