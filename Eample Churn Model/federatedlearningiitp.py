import dill as pickle
import socket
from _thread import *
import threading
import sys
import time
import os
import tensorflow as tf
import numpy as np

client_subscribed = []
weights_set = []
last_client = -1
connections = 0
status = ""

register_client_lock = threading.Lock()
register_weights_lock = threading.Lock()
new_weights_lock = threading.Lock()
persistentTCP_lock = threading.Lock()

class FederatedCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		global signal
		global status
		if signal:
			print("Training Interrupted")
			self.model.stop_training = True

def greetServer(s):
	s = connect(host, port)
	s.sendall(pickle.dumps("HELLO"))
	msg = pickle.loads(s.recv(1024))
	return msg

def subscribeServer(host, port):
	s = connect(host, port)
	s.sendall(pickle.dumps("SUBSCRIBE"))

def requestModelGen(host, port):
	s = connect(host, port)
	s.sendall(pickle.dumps("REQUEST MODEL FUNCTION"))
	size = pickle.loads(s.recv(1024))
	s.sendall(pickle.dumps("OK"))
	msg = pickle.loads(recvall(s,size))
	return msg

def requestPreprocessing(host, port):
	s = connect(host, port)
	s.sendall(pickle.dumps("REQUEST PREPROCESSING"))
	size = pickle.loads(s.recv(1024))
	s.sendall(pickle.dumps("OK"))
	msg = pickle.loads(recvall(s,size))
	return msg

def requestDataCleaner(host, port):
	s = connect(host, port)
	s.sendall(pickle.dumps("REQUEST DATA CLEANER"))
	size = pickle.loads(s.recv(1024))
	s.sendall(pickle.dumps("OK"))
	msg = pickle.loads(recvall(s,size))
	return msg

def requestWeights(host, port):
	s = connect(host, port)
	s.sendall(pickle.dumps("REQUEST WEIGHTS"))
	size = pickle.loads(s.recv(1024))
	s.sendall(pickle.dumps("OK"))
	msg = pickle.loads(recvall(s,size))
	return msg

def sendWeights(host, port, weights):
	s = connect(host, port)
	s.sendall(pickle.dumps("SENDING WEIGHTS"))
	ack = pickle.loads(s.recv(1024))
	if ack == "OK":
		sendObject(s,weights)


def aggregate(weights_set):
	new_weights = [np.zeros(w.shape) for w in weights_set[0]]
	for w in weights_set:
		for i in range(len(new_weights)):
			new_weights[i] += w[i]/len(weights_set)	
	return new_weights

def evaluate(getModel,X,y,accuracy_cutoff):
	model = getModel()
	model.set_weights(weights)
	results = model.evaluate(X, y)
	print("accuracy: "+str(results[1]))
	print("loss: "+str(results[0]))
	if results[1] > accuracy_cutoff:
		return "STOP"
	else:
		return "CONTINUE"

def updateClients(client_subscribed, status):
	persistentTCP_lock.acquire()
	for c in client_subscribed:
		c.sendall(pickle.dumps(status))
	persistentTCP_lock.release()


def recvall(s,size):
	fragments = []
	while size:
		chunk = s.recv(size)
		size -= len(chunk)
		fragments.append(chunk)
	return b''.join(fragments)

def sendObject(s,obj):
	pickled_obj = pickle.dumps(obj)
	s.sendall(pickle.dumps(len(pickled_obj)))
	ack = pickle.loads(s.recv(1024))
	if ack == "OK":
		s.sendall(pickled_obj)

def persistentTCP(host, port, max_clients):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
	s.bind((host, port))
	s.listen(max_clients)
	print("socket binded to port", port)
	while True:
		c, addr = s.accept()
		persistentTCP_lock.acquire()
		client_subscribed.append(c)
		persistentTCP_lock.release()

def communicate(c):
	global weights
	global last_client
	global weights_set
	global client_subscribed
	global aggregate_size
	global client_subscribed
	global connections
	global last_client
	connections += 1
	client_id = connections - 1
	register_client_lock.release() 
	msg = c.recv(1024)
	if msg:
		msg = pickle.loads(msg)
	if msg == "HELLO":
		c.sendall(pickle.dumps("HELLO"))
	elif msg == "REQUEST MODEL FUNCTION":
		sendObject(c,getModel)
	elif msg == "REQUEST PREPROCESSING":
		sendObject(c,preprocessing)
	elif msg == "REQUEST DATA CLEANER":
		sendObject(c,cleanData)
	elif msg == "REQUEST WEIGHTS":
		sendObject(c,weights)
	elif msg == "SENDING WEIGHTS":
		c.sendall(pickle.dumps("OK"))
		size = pickle.loads(c.recv(1024))
		if not new_weights_lock.locked():
			c.sendall(pickle.dumps("OK"))
			recv_weights = pickle.loads(recvall(c,size))
			register_weights_lock.acquire()
			weights_set.append(recv_weights)
			last_client = client_id
			register_weights_lock.release()
			if len(weights_set) >= aggregate_size:
				time.sleep(wait_time)
				if last_client == client_id:
					new_weights_lock.acquire()
					weights = aggregate(weights_set)
					weights_set = []
					status = evaluate(getModel,X,y,accuracy_cutoff)
					updateClients(client_subscribed, status)
					new_weights_lock.release()
					if status == "STOP":
						for c in client_subscribed:
							c.close()
						print("Threshhold Reached. Training Stopped.")

		else:
			c.sendall(pickle.dumps("BUSY"))	
	else:
		c.sendall(pickle.dumps("REQUEST NOT ACCEPTED"))
	c.close()

def connect(host,port):
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
	s.connect((host,port))
	return s

def train(init_weights, X, y , epochs, batchsize, host, port,training_lock):
	global signal
	training_lock.acquire()
	signal = False
	model = getModel()
	model.set_weights(init_weights)
	model.fit(X, y, batch_size = batchsize, epochs = epochs, callbacks=[FederatedCallback()])
	print(model.evaluate(X,y))
	if not signal:
		sendWeights(host,port,model.get_weights())
	training_lock.release()

def startFederatedLearning(host, port, pport, weights, modelGen, X, y, epochs = 10, batchsize = 10):
	global signal
	global getModel
	getModel = modelGen
	training_lock = threading.Lock()
	persistentTCP = connect(host,pport)
	start_new_thread(train, (weights, X,y,epochs,batchsize,host,port,training_lock,))
	while True:
		msg = persistentTCP.recv(1024)
		if not msg:
			time.sleep(1)
			continue
		status = pickle.loads(msg)
		weights = requestWeights(host, port)
		signal = True
		if status == "STOP":
			return weights
		elif status == "CONTINUE":
			start_new_thread(train, (weights, X, y, epochs, batchsize, host, port, training_lock,))

def createServer(host, port, pport, max_clients, Raggregate_size, Rwait_time, Raccuracy_cutoff, RX, Ry, RgetModel, Rpreprocessing, RcleanData):
	global wait_time
	global aggregate_size
	global accuracy_cutoff
	global getModel
	global preprocessing
	global cleanData
	global weights
	global X
	global y
	X = RX
	y = Ry
	wait_time = Rwait_time
	aggregate_size = Raggregate_size
	accuracy_cutoff = Raccuracy_cutoff
	getModel = RgetModel
	preprocessing = Rpreprocessing
	cleanData = RcleanData
	weights = getModel().get_weights()
	print("At Start")
	evaluate(RgetModel,X,y,70)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
	s.bind((host, port)) 
	print("socket binded to port", port)
	s.listen(max_clients*100)
	start_new_thread(persistentTCP, (host ,pport, max_clients))
	while True:
		c, addr = s.accept()
		register_client_lock.acquire() 
		print('Connection ',connections,' established:', addr[0], ':', addr[1])
		start_new_thread(communicate, (c,))	
