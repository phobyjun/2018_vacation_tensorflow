import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

x = queue1.dequeue()
x.eval()

x.eval()

x.eval()

x.eval()

queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])

enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

inputs = queue1.dequeue_many(4)
inputs.eval()

sess.run(queue1.size())
