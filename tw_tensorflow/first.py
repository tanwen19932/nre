import numpy as np
import tw_tensorflow as tf

w = tf.variable(0)
cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.session
session.run(init)
print(session.run(w))
session.run(train)


print(session.run(w))
for i in range(1000):
    session.run(train)
print(session.run(w))