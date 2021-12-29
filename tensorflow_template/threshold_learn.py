import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
graph = tf.Graph()
with graph.as_default():
    input_t = tf.placeholder(tf.float32, [2,3], 'input')
    ys = tf.placeholder(dtype=tf.float32, shape=(None))
    threshold_t = tf.Variable(0.9)

    fc1= Dense(100, activation='relu')
    fc2= Dense(3, activation='softmax')
    out_t = fc2(fc1(tf.maximum(input_t, threshold_t)))

    cross_entropy = tf.losses.softmax_cross_entropy(ys, out_t)  # ys size [batch_size, 10]

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        print('out', sess.run(train_step, feed_dict={input_t: [[-0.3, 0.0, 0.7],[-0.3, 0.0, 0.7]], ys:[0.1, 0.3, 0.5]}))
        print(threshold_t)
        # get grad of out_t wrt threshold_t
        grad_out_t = tf.gradients(cross_entropy, [threshold_t])[0]
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.0, 0.7],[-0.3, 0.0, 0.7]], ys:[0.1, 0.3, 0.5]}))
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.0, 0.7],[-0.3, 0.0, 0.7]], ys:[0.1, 0.3, 0.2]}))
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-100, 0.0, 0.7],[-0.3, 0.0, 7]], ys:[0.1, 0.3, 0.5]}))
        # print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.0, -0.7],[-0.3, 0.0, 0.7]]}))
        # print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.5, 0.7],[-0.3, 0.0, 0.7]]}))