def build_graph(FLAGS):
  """Define training graph.
  """
  with tf.device(FLAGS.device):
    # Graph input
    x = tf.placeholder(tf.float32, shape=(None, None, FLAGS.n_input)) # (seq_len, batch_size, n_input)
    actions = tf.placeholder(tf.float32, shape=(None, None)) # (seq_len, batch_size)
    advantage = tf.placeholder(tf.float32, shape=(None, None)) # (seq_len, batch_size)
    x_mask = tf.placeholder(tf.float32, shape=(None, None)) # (seq_len, batch_size)
    state = tf.placeholder(tf.float32, shape=(2, None, FLAGS.n_hidden))
    y = tf.placeholder(tf.int32, shape=(None, None)) # (seq_len, batch_size)

    one_step_x = tf.placeholder(tf.float32, shape=(None, FLAGS.n_input)) # (batch_size, n_input)
    one_step_state = tf.placeholder(tf.float32, shape=(2, None, FLAGS.n_hidden))

  # Define LSTM module
  _rnn = LSTMModule(FLAGS.n_hidden)
  # Call LSTM module
  one_step_c_state, one_step_h_state = _rnn(one_step_x, one_step_state, one_step=True)
  h_rnn_3d, last_state = _rnn(x, state)
  # Reshape into [seq_len*batch_size, num_units]
  h_rnn_2d = tf.reshape(h_rnn_3d, [-1, FLAGS.n_hidden])
  # Define output layer
  _output = LinearCell(FLAGS.n_class, scope='output_module')
  # Call output layer [seq_len*batch_size, n_class]
  h_logits = _output(h_rnn_2d, 'output')
  one_step_h_output_logits = _output(one_step_h_rnn)
  one_step_h_output_prob = tf.softmax(one_step_h_output_logits, axis=1)
  # Transform labels into one-hot vectors [seq_len*batch_size, n_class]
  y_1hot = tf.one_hot(tf.reshape(y, [-1]), depth=FLAGS.n_class)
  # Skip cell
  _skip_output_logits = LinearCell(FLAGS.n_class, scope='skip_module')
  one_step_h_skip_output_logits = _skip_output_logits(one_step_h_rnn)
  one_step_h_skip_output_prob = tf.softmax(one_step_h_skip_output_logits, axis=1)

  h_skip_output_logits = _skip_output_logits(h_rnn_2d) 
  h_skip_output_prob = tf.softmax(h_skip_output_logits, axis=1)
  # Define ML objective, size = [seq_len*batch_size]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_1hot,
                                                          logits=h_logits)
  # Define REINFORCE objective
  # REINFOCE_cost = log(prob) * Advantage
  reinforce_cost = categorical_log_prob(h_skip_output_prob, actions) * advantage
  # REINFORCE PART
  # h_skip_output_logits = _skip_output_logits(h_rnn_2d)
  # h_skip_output_prob

  ml_cost = tf.reduce_sum((cross_entropy * tf.reshape(x_mask, [-1])), axis=0)
  rl_cost = tf.reduce_sum((reinforce_cost * tf.reshape(x_mask, [-1])), axis=0)
  cost = ml_cost + rl_cost
  return Graph(cost, x, x_mask, state, last_state, y), OneStepGraph(one_step_c_state, one_step_h_state, one_step_h_skip_output_prob, one_step_h_output_prob)
