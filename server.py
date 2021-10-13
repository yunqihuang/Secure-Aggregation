import numpy as np
import eventlet
import socketio
import base64
from enum import Enum
from sslib import shamir, randomness
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import random


class States(Enum):
    READY = 0
    BEGIN = 1
    FINISH = 2


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


# data structure maintain models and information about users
# can be replaced by database
class SecAggServer:
    def __init__(self):
        self.clientSecrets = {}
        self.cipherList = []
        self.clientU1_info = []
        self.U1set = []
        self.U2set = []
        self.U3set = []
        self.U4set = []
        self.models = []
        self.res = np.zeros(10)
        self.phase = {
            'U1': States.READY,
            'U2': States.READY,
            'U3': States.READY,
            'U4': States.READY,
        }


def create_serve(s, n, threshold):
    sio = socketio.Server()
    app = socketio.WSGIApp(sio, static_files={
        '/': {'content_type': 'text/html', 'filename': 'index.html'}
    })
    print('start server')

    # Background task to handle the timeout situation of Round1: 'AdvertiseKeys'
    def TimeoutU1():
        timeout = 5
        i = 0
        while i < timeout and s.phase['U1'] != States.FINISH:
            eventlet.sleep(1)
            i += 1
        print('U1 num is', len(s.clientU1_info))
        if len(s.U1set) < threshold:
            sio.emit('disconnect')
        sio.emit('clientU1', s.clientU1_info)

    # Background task to handle the timeout situation of Round2: 'ShareKeys'
    def TimeoutU2():
        timeout = 5
        i = 0
        while i < timeout and s.phase['U2'] != States.FINISH:
            eventlet.sleep(1)
            i += 1
        if len(s.U2set) < threshold:
            sio.emit('disconnect')
        for cid in s.U2set:
            res = []
            for item in s.cipherList:
                tid = item['id']
                for cipher in item['cipher']:
                    if cipher['id'] == cid:
                        res.append({'id': tid, 'cipher': cipher['cipher']})
            sio.emit('clientU2', res, room=cid)

    # Background task to handle the timeout situation of Round3: 'postModels'
    def TimeoutU3():
        timeout = 6
        i = 0
        while i < timeout and s.phase['U3'] != States.FINISH:
            eventlet.sleep(1)
            i += 1
        print(s.res)
        if len(s.U3set) < threshold:
            sio.emit('disconnect')
        sio.emit('clientU3', s.U3set)

    # Background task to handle the timeout situation of Round4: 'Unmasking'
    def TimeoutU4():
        timeout = 5
        i = 0
        while i < timeout and s.phase['U4'] != States.FINISH:
            eventlet.sleep(1)
            i += 1
        if len(s.U4set) < threshold:
            sio.emit('disconnect')
        else:
            for user in s.U2set:
                if user in s.U3set:
                    tmp = s.clientSecrets[user]['buShares']
                    bu = shamir.recover_secret(shamir.from_base64(tmp))
                    random.seed(bu)
                    for i in range(len(s.res)):
                        s.res[i] -= random.random()
                else:
                    tmp = s.clientSecrets[user]['suShares']
                    suSK = serialization.load_pem_private_key(
                        shamir.recover_secret(shamir.from_base64(tmp)),
                        None,
                        default_backend())
                    x = 1
                    for c in s.clientU1_info:
                        if c['id'] == user:
                            x = -1
                            continue
                        if c['id'] in s.U2set:
                            svPK = serialization.load_pem_public_key(c['suPK'].encode('utf-8'),
                                                                     default_backend())
                            shareKey = deriveKey(suSK, svPK)
                            random.seed(shareKey)
                            for i in range(len(s.res)):
                                s.res[i] += random.random() * x
            print(s.res)
            sio.emit('success')

    @sio.event
    def connect(sid, environ):
        print('connect ', sid)
        sio.emit('connect_success', {'id': sid}, room=sid)

    #  round 1: collect user uploaded suPk and cuPk
    # start the background task TimeoutU1, when meeting the requirement emit client message to every user
    @sio.on('AdvertiseKeys')
    def on_uploadPk(sid, data):
        if s.phase['U1'] == States.READY:
            s.phase['U1'] = States.BEGIN
            sio.start_background_task(TimeoutU1)
        data['id'] = sid
        print('server received Pk from {}'.format(sid))
        s.clientU1_info.append(data)
        s.U1set.append(sid)
        assert (len(s.U1set) == len(s.clientU1_info))
        if len(s.U1set) == n:
            s.phase['U1'] = States.FINISH
            # sio.emit('clientU1', s.clientU1_info)

    #  round 2: collect user uploaded encrypted shares
    # start the background task TimeoutU1, when meeting the requirement emit encrypted shares to every user
    @sio.on('ShareKeys')
    def on_secrets(sid, data):
        if s.phase['U2'] == States.READY:
            s.phase['U2'] = States.BEGIN
            sio.start_background_task(TimeoutU2)
        print('receive secrets from {}'.format(sid))
        cBuShares = {
            'required_shares': threshold,
            'prime_mod': data['buPrimeMod'],
            'shares': []
        }
        cSuShares = {
            'required_shares': threshold,
            'prime_mod': data['suSKPrimeMod'],
            'shares': []
        }
        s.U2set.append(sid)
        s.clientSecrets[sid] = {'buShares': cBuShares, 'suShares': cSuShares}
        s.cipherList.append({'id': sid, 'cipher': data['ciphertexts']})
        assert (len(s.U2set) == len(s.cipherList))
        if len(s.U2set) == n:
            s.phase['U2'] = States.FINISH

    # round 3: collect masked models from users
    @sio.on('postModels')
    def on_postModels(sid, data):
        if s.phase['U3'] == States.READY:
            s.phase['U3'] = States.BEGIN
            sio.start_background_task(TimeoutU3)
        print(str(sid) + ' post models')
        s.U3set.append(sid)
        r = base64.b64decode(data)
        model = np.frombuffer(r, dtype=np.dtype('d'))
        s.models.append({'id': sid, 'model': model})
        assert (len(s.U3set) == len(s.models))
        s.res += model
        if len(s.U3set) == n:
            s.phase['U3'] = States.FINISH

    # round 4: collect decrypted bu shares or suSk shares of user, then unmask the model
    @sio.on('Unmasking')
    def on_Unmasking(sid, data):
        if s.phase['U4'] == States.FINISH:
            return
        if s.phase['U4'] == States.READY:
            s.phase['U4'] = States.BEGIN
            sio.start_background_task(TimeoutU4)
        print('receive shares from {}'.format(sid))
        for share in data:
            if share['buShare'] is not None:
                tmp = s.clientSecrets[share['id']]['buShares']
                tmp['shares'].append(share['buShare'])
                s.clientSecrets[share['id']]['buShares'] = tmp
            if share['suShare'] is not None:
                tmp = s.clientSecrets[share['id']]['suShares']
                tmp['shares'].append(share['suShare'])
                s.clientSecrets[share['id']]['suShares'] = tmp
        s.U4set.append(sid)
        if len(s.U4set) == threshold + 1:
            s.phase['U4'] = States.FINISH

    @sio.event
    def disconnect(sid):
        print('disconnect ', sid)

    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)


server = SecAggServer()
create_serve(server, 5, 2)
