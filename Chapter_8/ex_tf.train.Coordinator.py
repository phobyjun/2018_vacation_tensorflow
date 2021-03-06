import tensorflow as tf
import threading
import time

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

sess = tf.InteractiveSession()

def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        if i == 11:
            coord.request_stop()

coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
