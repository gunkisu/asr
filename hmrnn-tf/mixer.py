import math
import numpy as np
import scipy.signal
import tensorflow as tf
import itertools

from tensorflow.python.framework import ops


# Misc
def gen_mask(x, max_seq_len):
  n_step = x.shape[0]
  if n_step != max_seq_len:
    padded_x = np.zeros((max_seq_len, x.shape[1]))
    padded_x[:n_step] = x
    x = padded_x.astype('int64')
    x_mask = np.zeros((max_seq_len, x.shape[1])).astype(np.float32)
    x_mask[:n_step] = 1.
  else:
    x_mask = np.ones((x.shape[0], x.shape[1])).astype(np.float32)
  return x, x_mask


def insert_item2dict(ref_dict, key, value):
  try:
    ref_dict[key].append(value)
  except:
    ref_dict[key] = [value]


def nats2bits(x):
  return x / math.log(2)


def save_npz2(file_name, param_dict):
  np.savez(file_name, **param_dict)


# Initialization
def custom_init(nin, nout=None, scale=0.01, orthogonal=True):
  if nout is None:
    nout = nin
  if nout == nin and orthogonal:
    x = orthogonal_init(nin)
  else:
    x = glorot_init(nin, nout)
  return x


def glorot_init(nin, nout=None, uniform=True):
  if nout is None:
    nout = nin
  if uniform:
    scale = np.sqrt(6.0 / (nin + nout))
    x = uniform_init(nin, nout, scale)
  else:
    scale = np.sqrt(3.0 / (nin + nout))
    x = normal_init(nin, nout, scale)
  return x


def he_init(nin, nout=None, uniform=True):
  if nout is None:
    nout = nin
  scale = np.sqrt(1.0 / nin)
  if uniform:
    x = uniform_init(nin, nout, scale)
  else:
    x = normal_init(nin, nout, scale)
  return x


def normal_init(nin, nout=None, scale=0.01):
  x = scale * np.random.normal(loc=0.0, scale=1.0, size=(nin, nout))
  return x.astype(np.float32)


def orthogonal_init(nin):
  x = np.random.normal(0.0, 1.0, (nin, nin))
  u, _, v = np.linalg.svd(x, full_matrices=False)
  return u.astype(np.float32)


def uniform_init(nin, nout=None, scale=0.01):
  x = np.random.uniform(size=(nin, nout), low=-scale, high=scale)
  return x.astype(np.float32)


# Tensorflow
def add_summary_value(summary, tag, simple_value):
  value = summary.value.add()
  value.tag = tag
  value.simple_value = simple_value


# Tensorflow Activation
def hard_sigmoid(x, scale=1.):
  return tf.clip_by_value((scale * x + 1.) / 2., clip_value_min=0,
                          clip_value_max=1)


# Tensorflow Op
def binary_round(x):
  """Rounds a tensor whose values are in [0, 1] to a tensor
  with values in {0, 1}, using the straight through estimator
  to approximate the gradient.
  """
  g = tf.get_default_graph()
  with ops.name_scope("binary_round") as name:
    with g.gradient_override_map({"Round": "Identity"}):
      return tf.round(x, name=name)


def binary_sigmoid(x, slope_tensor=None):
  """Straight through hard sigmoid.
  Hard sigmoid followed by the step function.
  """
  if slope_tensor is None:
    slope_tensor = tf.constant(1.0)
  p = hard_sigmoid(x, slope_tensor)
  return binary_round(p)


def scatter_add_tensor(ref, indices, updates, name=None):
  """
  Adds sparse updates to a variable reference.

  This operation outputs ref after the update is done. This makes it easier
  to chain operations that need to use the reset value.

  Duplicate indices are handled correctly: if multiple indices reference
  the same location, their contributions add.

  Requires updates.shape = indices.shape + ref.shape[1:].
  Args:
    param ref: A Tensor. Must be one of the following types: float32, float64,
      int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8,
      quint8, qint32, half.
    param indices: A Tensor. Must be one of the following types: int32, int64.
      A tensor of indices into the first dimension of ref.
    param updates: A Tensor. Must have the same dtype as ref.
      A tensor of updated values to add to ref
    param name: A name for the operation (optional).
  Return:
    Same as ref. Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
  with tf.name_scope(name, 'scatter_add_tensor',
                     [ref, indices, updates]) as scope:
    ref = tf.convert_to_tensor(ref, name='ref')
    indices = tf.convert_to_tensor(indices, name='indices')
    updates = tf.convert_to_tensor(updates, name='updates')
    ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
    scattered_updates = tf.scatter_nd(indices, updates, ref_shape,
                                      name='scattered_updates')
    with tf.control_dependencies(
            [tf.assert_equal(ref_shape,
                             tf.shape(scattered_updates,
                                      out_type=indices.dtype))]):
      output = tf.add(ref, scattered_updates, name=scope)
  return output


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

class LinearVF(object):
#    def __init__(self, reg_coeff=2.0, num_iter=1):
    def __init__(self, reg_coeff=2.0, num_iter=5):
        self.coeffs = None
#        self.reg_coeff = 2.0
        self.reg_coeff = 1e-5
        self.num_iter = num_iter

    def _features(self, X):
        o = X.astype('float32')
        return np.concatenate([o, o**2, o**3])

    def get_featmat(self, X):
        return np.asarray([self._features(x) for x in X])

    def fit(self, X, returns):
        featmat = self.get_featmat(X)
        reg_coeff = self.reg_coeff
        for _ in range(self.num_iter):
            # Equation 3.28 in PRML
            self.coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self.coeffs)):
                break
#            reg_coeff *= 2
            reg_coeff *= 10

    def predict(self, X):
        if self.coeffs is None: 
            return np.zeros(X.shape[0])
        return self.get_featmat(X).dot(self.coeffs)

def skip_rnn_act(x, x_mask, y, sess, sample_graph, args):
    """Sampling episodes using Skip-RNN"""

    # x shape is [time_step, batch_size, features]
    n_seq, n_batch, x_size = x.shape
    act_size = args.n_action
    reshape_x = np.transpose(x, [1, 0, 2])
    reshape_y = np.transpose(y, [1, 0])
    batch_size = len(reshape_x)

    new_X = []
    new_Y = []
    actions = []
    rewards = []
    action_entropies = []

    # recording
    full_action_samples = np.zeros([n_seq, n_batch, act_size])
    full_action_probs = np.zeros([n_seq, n_batch, act_size])

    # for each data (index i)
    for i, (x_i, y_i) in enumerate(zip(reshape_x, reshape_y)):
        prev_state = np.zeros((2, 1, args.n_hidden))
        new_x_i = []
        new_y_i = []
        action_i = []
        reward_i = []
        action_entropy_i = []
        action_cnt = 0    
        # last action is ignored in this implementation
        # find a way to better handle the last action sampling

        # for each step (index j)
        for j, (x_step, y_step) in enumerate(zip(x_i, y_i)):
            # for the last step
            if j == len(x_i) - 1:
                x_step = np.expand_dims(x_step, 0)
                [step_label_likelihood_j,
                 prev_state]  = sess.run([sample_graph.step_label_probs,
                                          sample_graph.step_last_state],
                                          feed_dict={sample_graph.step_x_data: x_step,
                                                     sample_graph.prev_states: prev_state})
                new_x_i.append(x_step)
                new_y_i.append(y_step)
                reward_i.append(np.log(step_label_likelihood_j.flatten()[y_step] + 1e-8))
            else:
                # if action is required
                if action_cnt == 0:
                    x_step = np.expand_dims(x_step, 0)
                    [step_action_idx,
                     step_action_prob_j,
                     step_label_likelihood_j,
                     prev_state,
                     action_entropy] = sess.run([sample_graph.step_action_samples,
                                                 sample_graph.step_action_probs,
                                                 sample_graph.step_label_probs,
                                                 sample_graph.step_last_state,
                                                 sample_graph.action_entropy],
                                                feed_dict={sample_graph.step_x_data: x_step,
                                                           sample_graph.prev_states: prev_state})
                    new_x_i.append(x_step)
                    new_y_i.append(y_step)
                    action_entropy_i.append(action_entropy)

                    action_idx = step_action_idx.flatten()
                    action_one_hot = np.eye(args.n_action)[action_idx]
                    action_i.append(action_one_hot)

                    if args.fast_action and action_idx == args.n_action - 1:
                        # action in {0, 1, 2, ... fast_action}
                        action_cnt = args.n_fast_action
                    else:
                        # action in {0, 1, 2, ...}
                        action_cnt = action_idx + 1

                    action_cnt -= 1
                    if j != 0:
                        reward_i.append(np.log(step_label_likelihood_j.flatten()[y_step] + 1e-8))

                    # record
                    full_action_samples[j, i, action_idx] = 1.0
                    full_action_probs[j, i] = step_action_prob_j

                else:
                    # skip frame
                    action_cnt -= 1

        new_X.append(new_x_i)
        new_Y.append(new_y_i)
        actions.append(action_i)
        rewards.append(reward_i)
        action_entropies.append(action_entropy_i)

    # masking episodes
    new_masked_X, new_masked_Y, masked_actions, masked_rewards, masked_action_entropies, new_mask, new_reward_mask = \
        mask_episodes(new_X, new_Y, actions, rewards, action_entropies, batch_size, x_size, act_size)

    # Make visual image
    full_action_samples = np.transpose(full_action_samples, [1, 2, 0])
    full_action_samples = np.expand_dims(full_action_samples, axis=-1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=2)

    full_action_probs = np.transpose(full_action_probs, [1, 2, 0])
    full_action_probs = np.expand_dims(full_action_probs, axis=-1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=2)

    # batch_size, seq_len
    full_label_data = np.expand_dims(y, axis=-1)
    full_label_data = np.transpose(full_label_data, [1, 2, 0])
    full_label_data = np.expand_dims(full_label_data, axis=-1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=2).astype(np.float32)
    full_label_data /= float(args.n_class)

    # stack
    output_image = np.concatenate([np.concatenate([full_label_data,
                                                   np.zeros_like(full_label_data),
                                                   np.zeros_like(full_label_data)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_samples),
                                                   full_action_samples,
                                                   np.zeros_like(full_action_samples)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_probs),
                                                   np.zeros_like(full_action_probs),
                                                   full_action_probs], axis=-1)],
                                  axis=1)
        
    return (new_masked_X, new_masked_Y, masked_actions, masked_rewards, masked_action_entropies, new_mask, new_reward_mask, output_image)

def filter_last(x_step, y_step, prev_state, j, seq_lens, sample_done):
    new_x_step = []
    new_y_step = []
    new_prev_state = []
    target_indices = [] 

    for i, (x, y, l, p) in enumerate(itertools.izip(x_step, y_step, seq_lens, prev_state)):
        if i in sample_done: continue 

        if j == l-1:
            new_x_step.append(x)
            new_y_step.append(y)
            new_prev_state.append(p)
            target_indices.append(i)
    
    return np.asarray(new_x_step), np.asarray(new_y_step), \
        np.asarray(new_prev_state), target_indices

def filter_action_end(x_step, y_step, prev_state, j, action_counters, sample_done):
    new_x_step = []
    new_y_step = []
    new_prev_state = []
    target_indices = [] 

    for i, (x, y, p, ac) in enumerate(zip(x_step, y_step, prev_state, action_counters)):
        if i in sample_done: continue 

        if ac == 0:
            new_x_step.append(x)
            new_y_step.append(y)
            new_prev_state.append(p)
            target_indices.append(i)

    return np.asarray(new_x_step), np.asarray(new_y_step), \
        np.asarray(new_prev_state), target_indices

def fill(x, x_step, target_indices, update_pos):
    for x_value, idx in zip(x_step, target_indices):
        x[update_pos[idx], idx] = x_value

def fill_reward(rewards, reward_step, target_indices, reward_update_pos, ref_update_pos):
    reward_target_indices = []
    for r, idx in zip(reward_step, target_indices):
        if ref_update_pos[idx] == 0:
            continue

        rewards[reward_update_pos[idx], idx] = r
        reward_target_indices.append(idx)

    return reward_target_indices


def fill_aggr_reward(reward_list,
                     y_seq,
                     cur_pred_idx_list,
                     cur_step_idx,
                     prev_pred_idx_list,
                     prev_step_idx_list,
                     target_indices,
                     reward_update_pos,
                     ref_update_pos):
    reward_target_indices = []

    # For each sample
    for i, idx in enumerate(target_indices):
        # If current action is first
        if ref_update_pos[idx] == 0:
            continue

        # Get previous action info
        prev_step_idx = prev_step_idx_list[idx]
        prev_pred_idx = prev_pred_idx_list[idx]

        # Get action size
        action_size = cur_step_idx - prev_step_idx

        # Get true label from previous action position to now
        true_label = y_seq[prev_step_idx:cur_step_idx, idx]

        cnt_seg_len = 0.
        seg_idx = true_label[0]
        for j in true_label:
            if j == seg_idx:
                cnt_seg_len += 1.
            else:
                break

        if cnt_seg_len == action_size:
            aggr_reward = action_size
        else:
            aggr_reward = 0.0
        aggr_reward = np.square(aggr_reward)

        # # Get prediction label (copy from previous action) and current prediction label
        # pred_label = [prev_pred_idx] * (cur_step_idx - prev_step_idx) + [cur_pred_idx_list[i]]
        # pred_label = np.asarray(pred_label)
        #
        # # Aggregate reward
        # true_label = (true_label[1:]-true_label[:-1])!=0
        # pred_label = (pred_label[1:]-pred_label[:-1])!=0

        # aggr_reward = np.equal(true_label, pred_label).sum().astype(np.float32)
        # aggr_reward = np.square(aggr_reward)

        # Save rewards
        reward_list[reward_update_pos[idx], idx] = aggr_reward
        reward_target_indices.append(idx)

    return reward_target_indices

def advance_pos(update_pos, target_indices):
    for i in target_indices:
        update_pos[i] += 1

def update_prev_state(prev_state, new_prev_state, target_indices):
    for ps, i in zip(new_prev_state, target_indices):
        prev_state[i] = ps

def update_action_counters(action_counters, action_idx, target_indices, args):
    new_ac = list(action_counters)
    for ai, i in zip(action_idx, target_indices):
        if args.fast_action and ai == args.n_action - 1:
            new_ac[i] = args.n_fast_action
        else:
            new_ac[i] = ai+1

    # proceed to the next step
    new_ac = [ac-1 for ac in new_ac]
    action_counters[:] = new_ac

def gen_mask(update_pos, reward_update_pos, batch_size):
    max_seq_len = max(update_pos)
    max_reward_seq_len = max(reward_update_pos)
    mask = np.zeros([max_seq_len, batch_size])
    reward_mask = np.zeros([max_seq_len-1, batch_size])

    for i, pos in enumerate(update_pos):
        mask[:pos, i] = 1.

    for i, pos in enumerate(reward_update_pos):
        reward_mask[:pos, i] = 1.

    return max_seq_len, mask, max_reward_seq_len, reward_mask

def aggr_skip_rnn_act_parallel(x,
                               x_mask,
                               y,
                               sess,
                               sample_graph,
                               args):
    def transpose_all(new_x,
                      new_y,
                      actions,
                      rewards,
                      action_entropies,
                      new_x_mask,
                      new_reward_mask):
        return [np.transpose(new_x, [1,0,2]),
                np.transpose(new_y, [1,0]),
                np.transpose(actions, [1,0,2]),
                np.transpose(rewards, [1,0]),
                np.transpose(action_entropies, [1,0]),
                np.transpose(new_x_mask, [1,0]),
                np.transpose(new_reward_mask, [1,0])]

    """Sampling episodes using Skip-RNN"""

    # x shape is [time_step, batch_size, features]
    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0)
    max_seq_len = int(max(seq_lens))

    # shape should be (2, n_batch, n_hidden) when it is used
    prev_state = np.zeros((n_batch, 2, args.n_hidden))

    # init counter and positions
    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    reward_update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed
    prev_action_idx = [-1]*n_batch
    prev_action_pos = [-1]*n_batch

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])
    actions = np.zeros([max_seq_len-1, n_batch, args.n_action])
    rewards = np.zeros([max_seq_len-1, n_batch])
    action_entropies = np.zeros([max_seq_len-1, n_batch])

    # recording
    full_action_samples = np.zeros([max_seq_len, n_batch, args.n_action])
    full_action_probs = np.zeros([max_seq_len, n_batch, args.n_action])

    # for each step (index j)
    for j, (x_step, y_step) in enumerate(itertools.izip(x, y)):
        # Get final step data
        [_x_step,
         _y_step,
         _prev_state,
         target_indices] = filter_last(x_step,
                                       y_step,
                                       prev_state,
                                       j,
                                       seq_lens,
                                       sample_done)

        # If final step sample exists,
        if len(_x_step):
            # Read and update state
            [step_label_likelihood_j,
             new_prev_state] = sess.run([sample_graph.step_label_probs,
                                         sample_graph.step_last_state],
                                        feed_dict={sample_graph.step_x_data: _x_step,
                                                   sample_graph.prev_states: np.transpose(_prev_state, [1, 0, 2])})
            # prediction index
            step_label_idx = step_label_likelihood_j.argmax(axis=1)

            # Roll state
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])

            # Fill read data to new sequence
            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)

            # Set reward for previous actions
            reward_target_indices = fill_aggr_reward(rewards,
                                                     y,
                                                     step_label_idx,
                                                     j,
                                                     prev_action_idx,
                                                     prev_action_pos,
                                                     target_indices,
                                                     reward_update_pos,
                                                     update_pos)

            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)
            sample_done.extend(target_indices)

        # Based on action, get related samples
        [_x_step,
         _y_step,
         _prev_state,
         target_indices] = filter_action_end(x_step,
                                             y_step,
                                             prev_state,
                                             j,
                                             action_counters,
                                             sample_done)

        # If sample exist, process
        if len(_x_step):
            # Given input, update state and also action sample
            [action_idx,
             step_action_prob_j,
             step_label_likelihood_j,
             new_prev_state,
             action_entropy] = sess.run([sample_graph.step_action_samples,
                                         sample_graph.step_action_probs,
                                         sample_graph.step_label_probs,
                                         sample_graph.step_last_state,
                                         sample_graph.action_entropy],
                                        feed_dict={sample_graph.step_x_data: _x_step,
                                                   sample_graph.prev_states: np.transpose(_prev_state, [1, 0, 2])})
            # prediction index
            step_label_idx = step_label_likelihood_j.argmax(axis=1)

            # roll state
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])
            action_one_hot = np.eye(args.n_action)[action_idx.flatten()]

            # fill read data
            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(action_entropies, action_entropy, target_indices, update_pos)
            fill(actions, action_one_hot, target_indices, update_pos)

            # update counter
            update_action_counters(action_counters, action_idx.flatten(), target_indices, args)

            # Set reward for previous actions
            reward_target_indices = fill_aggr_reward(rewards,
                                                     y,
                                                     step_label_idx,
                                                     j,
                                                     prev_action_idx,
                                                     prev_action_pos,
                                                     target_indices,
                                                     reward_update_pos,
                                                     update_pos)

            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)

            # Set current prediction
            for label, idx in zip(step_label_idx, target_indices):
                prev_action_idx[idx] = label
                prev_action_pos[idx] = j

            # Save status for visualization
            for i, s_idx in enumerate(target_indices):
                # Set action sample
                full_action_samples[j, s_idx] = action_one_hot[i]

                # Set action prob
                full_action_probs[j, s_idx] = step_action_prob_j[i]
        else:
            update_action_counters(action_counters, [], [], args)

    max_seq_len, mask, max_reward_seq_len, reward_mask = gen_mask(update_pos, reward_update_pos, n_batch)

    # Make visual image
    full_action_samples = np.transpose(full_action_samples, [1, 2, 0])
    full_action_samples = np.expand_dims(full_action_samples, axis=-1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=2)

    full_action_probs = np.transpose(full_action_probs, [1, 2, 0])
    full_action_probs = np.expand_dims(full_action_probs, axis=-1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=2)

    # batch_size, seq_len
    full_label_data = np.expand_dims(y, axis=-1)
    full_label_data = np.transpose(full_label_data, [1, 2, 0])
    full_label_data = np.expand_dims(full_label_data, axis=-1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=2).astype(np.float32)
    full_label_data /= float(args.n_class)

    # stack
    output_image = np.concatenate([np.concatenate([full_label_data,
                                                   np.zeros_like(full_label_data),
                                                   np.zeros_like(full_label_data)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_samples),
                                                   full_action_samples,
                                                   np.zeros_like(full_action_samples)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_probs),
                                                   np.zeros_like(full_action_probs),
                                                   full_action_probs], axis=-1)],
                                  axis=1)
    return transpose_all(new_x[:max_seq_len],
                         new_y[:max_seq_len],
                         actions[:max_seq_len-1],
                         rewards[:max_reward_seq_len],
                         action_entropies[:max_seq_len-1],
                         mask,
                         reward_mask) + [output_image,]


def skip_rnn_act_parallel(x,
                          x_mask,
                          y,
                          sess,
                          sample_graph,
                          args):
    def transpose_all(new_x,
                      new_y,
                      actions,
                      rewards,
                      action_entropies,
                      new_x_mask,
                      new_reward_mask):
        return [np.transpose(new_x, [1,0,2]),
                np.transpose(new_y, [1,0]),
                np.transpose(actions, [1,0,2]),
                np.transpose(rewards, [1,0]),
                np.transpose(action_entropies, [1,0]),
                np.transpose(new_x_mask, [1,0]),
                np.transpose(new_reward_mask, [1,0])]

    """Sampling episodes using Skip-RNN"""

    # x shape is [time_step, batch_size, features]
    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0)
    max_seq_len = int(max(seq_lens))

    # shape should be (2, n_batch, n_hidden) when it is used
    prev_state = np.zeros((n_batch, 2, args.n_hidden))

    # init counter and positions
    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    reward_update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])
    actions = np.zeros([max_seq_len-1, n_batch, args.n_action])
    rewards = np.zeros([max_seq_len-1, n_batch])
    action_entropies = np.zeros([max_seq_len-1, n_batch])

    # recording
    full_action_samples = np.zeros([max_seq_len, n_batch, args.n_action])
    full_action_probs = np.zeros([max_seq_len, n_batch, args.n_action])

    # for each step (index j)
    for j, (x_step, y_step) in enumerate(itertools.izip(x, y)):
        # Get final step data
        [_x_step,
         _y_step,
         _prev_state,
         target_indices] = filter_last(x_step,
                                       y_step,
                                       prev_state,
                                       j,
                                       seq_lens,
                                       sample_done)

        # If final step sample exists,
        if len(_x_step):
            # Read and update state
            [step_label_likelihood_j,
             new_prev_state] = sess.run([sample_graph.step_label_probs,
                                         sample_graph.step_last_state],
                                        feed_dict={sample_graph.step_x_data: _x_step,
                                                   sample_graph.prev_states: np.transpose(_prev_state, [1, 0, 2])})

            # Roll state
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])

            # Fill read data to new sequence
            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)

            # Set reward for previous actions
            reward_target_indices = fill_reward(rewards,
                                                np.log(step_label_likelihood_j[range(len(_y_step)),_y_step] + 1e-8),
                                                target_indices,
                                                reward_update_pos,
                                                update_pos)

            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)
            sample_done.extend(target_indices)

        # Based on action, get related samples
        [_x_step,
         _y_step,
         _prev_state,
         target_indices] = filter_action_end(x_step,
                                             y_step,
                                             prev_state,
                                             j,
                                             action_counters,
                                             sample_done)

        # If sample exist, process
        if len(_x_step):
            # Given input, update state and also action sample
            [action_idx,
             step_action_prob_j,
             step_label_likelihood_j,
             new_prev_state,
             action_entropy] = sess.run([sample_graph.step_action_samples,
                                         sample_graph.step_action_probs,
                                         sample_graph.step_label_probs,
                                         sample_graph.step_last_state,
                                         sample_graph.action_entropy],
                                        feed_dict={sample_graph.step_x_data: _x_step,
                                                   sample_graph.prev_states: np.transpose(_prev_state, [1, 0, 2])})

            # roll state
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])
            action_one_hot = np.eye(args.n_action)[action_idx.flatten()]

            # fill read data
            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(action_entropies, action_entropy, target_indices, update_pos)
            fill(actions, action_one_hot, target_indices, update_pos)

            # update counter
            update_action_counters(action_counters, action_idx.flatten(), target_indices, args)

            # Set reward for previous actions
            reward_target_indices = fill_reward(rewards,
                                                np.log(step_label_likelihood_j[range(len(_y_step)), _y_step] + 1e-8),
                                                target_indices,
                                                reward_update_pos,
                                                update_pos)
            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)

            # Save status for visualization
            for i, s_idx in enumerate(target_indices):
                # Set action sample
                full_action_samples[j, s_idx] = action_one_hot[i]

                # Set action prob
                full_action_probs[j, s_idx] = step_action_prob_j[i]
        else:
            update_action_counters(action_counters, [], [], args)

    max_seq_len, mask, max_reward_seq_len, reward_mask = gen_mask(update_pos, reward_update_pos, n_batch)

    # Make visual image
    full_action_samples = np.transpose(full_action_samples, [1, 2, 0])
    full_action_samples = np.expand_dims(full_action_samples, axis=-1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=1)
    full_action_samples = np.repeat(full_action_samples, repeats=5, axis=2)

    full_action_probs = np.transpose(full_action_probs, [1, 2, 0])
    full_action_probs = np.expand_dims(full_action_probs, axis=-1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=1)
    full_action_probs = np.repeat(full_action_probs, repeats=5, axis=2)

    # batch_size, seq_len
    full_label_data = np.expand_dims(y, axis=-1)
    full_label_data = np.transpose(full_label_data, [1, 2, 0])
    full_label_data = np.expand_dims(full_label_data, axis=-1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=1)
    full_label_data = np.repeat(full_label_data, repeats=5, axis=2).astype(np.float32)
    full_label_data /= float(args.n_class)

    # stack
    output_image = np.concatenate([np.concatenate([full_label_data,
                                                   np.zeros_like(full_label_data),
                                                   np.zeros_like(full_label_data)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_samples),
                                                   full_action_samples,
                                                   np.zeros_like(full_action_samples)], axis=-1),
                                   np.concatenate([np.zeros_like(full_action_probs),
                                                   np.zeros_like(full_action_probs),
                                                   full_action_probs], axis=-1)],
                                  axis=1)
    return transpose_all(new_x[:max_seq_len],
                         new_y[:max_seq_len],
                         actions[:max_seq_len-1],
                         rewards[:max_reward_seq_len],
                         action_entropies[:max_seq_len-1],
                         mask,
                         reward_mask) + [output_image,]

def sample_from_softmax_batch(step_action_prob):
    action_one_hot = []
    action_idx = []

    for p in step_action_prob: 
        _action_one_hot = np.random.multinomial(1, p)
        action_one_hot.append(_action_one_hot)
        action_idx.append(np.argmax(_action_one_hot))
     
    return np.asarray(action_one_hot), np.asarray(action_idx)        

def sample_from_softmax(step_action_prob):
    action_one_hot = np.random.multinomial(1, step_action_prob.flatten())
    action_idx = np.argmax(action_one_hot)
     
    return action_one_hot, action_idx

def mask_episodes(X, Y, actions, rewards, action_entropies, batch_size, x_size, act_size):
#    X_size = X[0]
    max_time_step = max([len(x) for x in X])
    new_mask = np.zeros([batch_size, max_time_step])
    new_reward_mask = np.zeros([batch_size, max_time_step-1])
    masked_X = np.zeros([batch_size, max_time_step, x_size])
    masked_Y = np.zeros([batch_size, max_time_step])
    masked_actions = np.zeros([batch_size, max_time_step-1, act_size])
    masked_rewards = np.zeros([batch_size, max_time_step-1])
    masked_action_entropies = np.zeros([batch_size, max_time_step-1])

    for i, (x, y, action, reward, action_entropy) in enumerate(zip(X, Y, actions, rewards, action_entropies)):
        this_x_len = len(x)
        masked_X[i, :this_x_len, :] = x
        masked_Y[i, :this_x_len] = y
        masked_actions[i, :this_x_len-1, :] = action
        masked_rewards[i, :this_x_len-1] = reward
        masked_action_entropies[i, :this_x_len-1] = action_entropy
        new_mask[i, :this_x_len] = 1.
        new_reward_mask[i, :this_x_len-1] = 1.
   
    return masked_X, masked_Y, masked_actions, masked_rewards, masked_action_entropies, new_mask, new_reward_mask

def compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args):
    reward_mask_1d = new_reward_mask.reshape([-1])
    rewards_1d = rewards.reshape([-1])[reward_mask_1d==1.]
    discounted_rewards = []
    for reward, mask in zip(rewards, new_reward_mask):
        this_len = int(mask.sum())
        discounted_reward = discount(reward[:this_len], args.discount_gamma)
        discounted_rewards.append(discounted_reward)

    reshape_new_x = new_x.reshape([-1, new_x.shape[2]])
    baseline_1d = vf.predict(reshape_new_x)
    baseline_2d = baseline_1d.reshape([new_x.shape[0], -1]) * new_x_mask

    advantages = np.zeros_like(rewards)
    for i, (delta, mask) in enumerate(zip(baseline_2d, new_reward_mask)):
        this_len = int(mask.sum())
        advantages[i, :this_len] = discounted_rewards[i] - delta[:this_len]

    advantages_1d = advantages.reshape([-1])[reward_mask_1d==1.]
    advantages = ((advantages - advantages_1d.mean()) / (advantages_1d.std()+1e-8)) * new_reward_mask

    valid_x_indices= np.where(new_reward_mask==1.) 
    valid_new_x = new_x[valid_x_indices]
    discounted_rewards_1d = np.concatenate(discounted_rewards, axis=0)
    vf.fit(valid_new_x, discounted_rewards_1d)

    return advantages

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Shannon entropy for a paramaterized categorical distributions 
def categorical_ent(dist): 
    ent = -tf.reduce_sum(dist * tf.log(dist + 1e-8), axis=-1) 
    return ent

def expand_pred_idx(actions_1hot, x_mask, pred_idx, n_batch, args):
    new_pred_idx = np.zeros_like(x_mask)

    skip_info = np.argmax(actions_1hot, axis=2) + 1 # number of repeats
    pred_idx = pred_idx.reshape([n_batch, -1])

    # for each example
    for i, (s, p) in enumerate(zip(skip_info, pred_idx)):
        # for each step
        start_idx = 0
        for s_step, p_step in itertools.izip_longest(s, p, fillvalue=1):
            new_pred_idx[start_idx:start_idx+s_step, i] = p_step
            start_idx += s_step

    return new_pred_idx

def interpolate_feat(input_data, num_skips, axis=1, use_bidir=True):
    # Fill skipped ones by repeating (forward)
    full_fwd_data = np.repeat(input_data, repeats=num_skips, axis=axis)

    if use_bidir:
        full_bwd_data = np.repeat(np.flip(input_data, axis=axis), repeats=num_skips, axis=axis)
        full_bwd_data = np.flip(full_bwd_data, axis=axis)

        return (full_fwd_data + full_bwd_data)*0.5
    else:
        return full_fwd_data







