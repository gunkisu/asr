import math
import numpy as np
import scipy.signal
import itertools
import subprocess
import random

import tensorflow as tf

def gen_zero_state(n_batch, n_units):
    return np.zeros([n_batch, n_units], dtype=np.float32)

def feed_init_state(feed_dict, init_state, zero_state):
    for c, s in init_state:
        feed_dict.update({c: zero_state, s: zero_state})

def feed_prev_state(feed_dict, init_state, new_state):
    # shape of init_state: n_layer, 2, n_batch, n_hidden
    # shape of new state: n_batch, n_layer, 2, n_hidden
    # n_batch in init_state is None so it can take 
    # new_state in different batch sizes

    new_state = np.transpose(new_state, [1, 2, 0, 3])
    for i, n in zip(init_state, new_state):
        ic, ih = i
        nc, nh = n

        feed_dict.update({ic: nc, ih: nh})

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

def transpose_all(iterable):
    transposed = []

    for i in iterable:
        if len(i.shape) > 2:
            trans_i = np.transpose(i, [1,0,2])
        else:
            trans_i = np.transpose(i, [1,0])
        transposed.append(trans_i)
    
    return transposed

def insert_item2dict(ref_dict, key, value):
  try:
    ref_dict[key].append(value)
  except:
    ref_dict[key] = [value]


def nats2bits(x):
  return x / math.log(2)


def save_npz2(file_name, param_dict):
  np.savez(file_name, **param_dict)

class LinearVF(object):
#    def __init__(self, reg_coeff=2.0, num_iter=1):
    def __init__(self, reg_coeff=2.0, num_iter=5):
        self.coeffs = None
#        self.reg_coeff = 2.0
        self.reg_coeff = 1e-5
        self.num_iter = num_iter

    def _features(self, x):
        o = x.astype('float32')
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
        # [n_batch * n_seq, n_feat]
        
        if self.coeffs is None: 
            return np.zeros(X.shape[0]) # zeros of [n_batch * n_seq]

        return self.get_featmat(X).dot(self.coeffs)

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

def filter_last2(x_step, y_step, j, seq_lens, sample_done):
    new_x_step = []
    new_y_step = []
    target_indices = [] 

    for i, (x, y, l) in enumerate(itertools.izip(x_step, y_step, seq_lens)):
        if i in sample_done: continue 

        if j == l-1:
            new_x_step.append(x)
            new_y_step.append(y)
            target_indices.append(i)
    
    return np.asarray(new_x_step), np.asarray(new_y_step), target_indices

def filter_last_forward(x_step, prev_state, j, seq_lens, sample_done):
    new_x_step = []
    new_prev_state = []
    target_indices = [] 

    for i, (x, l, p) in enumerate(itertools.izip(x_step, seq_lens, prev_state)):
        if i in sample_done: continue 

        if j == l-1:
            new_x_step.append(x)
            new_prev_state.append(p)
            target_indices.append(i)
    
    return np.asarray(new_x_step), np.asarray(new_prev_state), target_indices


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

def filter_action_end2(x_step, y_step, j, action_counters, sample_done):
    new_x_step = []
    new_y_step = []
    target_indices = [] 

    for i, (x, y, ac) in enumerate(zip(x_step, y_step, action_counters)):
        if i in sample_done: continue 

        if ac == 0:
            new_x_step.append(x)
            new_y_step.append(y)
            target_indices.append(i)

    return np.asarray(new_x_step), np.asarray(new_y_step), target_indices

def filter_action_end_forward(x_step, prev_state, j, action_counters, sample_done):
    new_x_step = []
    new_prev_state = []
    target_indices = [] 

    for i, (x, p, ac) in enumerate(zip(x_step, prev_state, action_counters)):
        if i in sample_done: continue 

        if ac == 0:
            new_x_step.append(x)
            new_prev_state.append(p)
            target_indices.append(i)

    return np.asarray(new_x_step), np.asarray(new_prev_state), target_indices

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

def fill_seg_match_reward(reward_list, y, cur_step_idx, prev_pred_idx_list,
        prev_step_idx_list, target_indices, reward_update_pos, ref_update_pos, args):
    reward_target_indices = []

    for idx in target_indices:
        if ref_update_pos[idx] == 0:
            continue

        prev_step_idx = prev_step_idx_list[idx]
        prev_pred_idx = prev_pred_idx_list[idx]
        cur_skip = cur_step_idx - prev_step_idx
#        ref_labels = y[prev_step_idx:cur_step_idx, idx]

        max_jump = args.n_fast_action if args.n_fast_action > 0 else args.n_action
        ref_labels = y[prev_step_idx:prev_step_idx+max_jump, idx]
        
        match_count = 0
        miss_count = 0
        if args.use_prediction:
            target_label = prev_pred_idx
        else:
            target_label = ref_labels[0]

        for l in ref_labels:
            if l == target_label: match_count += 1
            else: break
      
        target_skip = match_count
        diff = target_skip - cur_skip
        # short skip
        if diff > 0:
            rw = -diff
        # target skip
        elif diff == 0:
            rw = 0
        # long skip
        else:
            rw = diff * args.alpha

#        reward_list[reward_update_pos[idx], idx] = rw - 1 # shifting
        reward_list[reward_update_pos[idx], idx] = rw
        reward_target_indices.append(idx)

    return reward_target_indices


def advance_pos(update_pos, target_indices):
    for i in target_indices:
        update_pos[i] += 1

def update_prev_state(prev_state, new_prev_state, target_indices=None):
    if target_indices:
        for ps, i in zip(new_prev_state, target_indices):
            prev_state[i] = ps
    else:
        for i, ps in enumerate(new_prev_state):
            prev_state[i] = ps


def update_action_counters2(action_counters, action_idx, target_indices, n_action, n_fast_action):
    new_ac = list(action_counters)
    for ai, i in zip(action_idx, target_indices):
        if n_fast_action > 0 and ai == n_action - 1:
            new_ac[i] = n_fast_action
        else:
            new_ac[i] = ai+1

    # proceed to the next step
    new_ac = [ac-1 for ac in new_ac]
    action_counters[:] = new_ac

def update_action_counters(action_counters, action_idx, target_indices, args):
    update_action_counters2(action_counters, action_idx, target_indices, args.n_action, args.n_fast_action)   

def gen_mask2(update_pos, reward_update_pos, batch_size):
    max_seq_len = max(update_pos)
    max_reward_seq_len = max(reward_update_pos)
    mask = np.zeros([max_seq_len, batch_size])
    reward_mask = np.zeros([max_seq_len-1, batch_size])

    for i, pos in enumerate(update_pos):
        mask[:pos, i] = 1.

    for i, pos in enumerate(reward_update_pos):
        reward_mask[:pos, i] = 1.

    return max_seq_len, mask, max_reward_seq_len, reward_mask

def gen_mask3(update_pos, batch_size):
    max_seq_len = max(update_pos)
    mask = np.zeros([max_seq_len, batch_size])

    for i, pos in enumerate(update_pos):
        mask[:pos, i] = 1.

    return max_seq_len, mask

def gen_mask_from(update_pos):
    batch_size = len(update_pos)
    max_seq_len = max(update_pos)
    mask = np.zeros([max_seq_len, batch_size])

    for i, pos in enumerate(update_pos):
        mask[:pos, i] = 1.

    return max_seq_len, mask

def color_flip(color, n_class):
    if color == n_class-1:
        return 0
    elif color == 0:
        return n_class-1
    else:
        raise ValueError

def to_label_change(y, n_class):
    # y is in shape [n_seq, n_batch]

    n_seq, n_batch = y.shape
    label_change = np.zeros([n_seq, n_batch])
    
    color = np.zeros([n_batch], dtype=np.int32)
    color[:] = n_class-1

    label_change[0] = color

    for i, (prev_color, cur_color) in enumerate(zip(y[:-1], y[1:]), start=1):
        for b, (pc, cc) in enumerate(zip(prev_color, cur_color)):
            if pc == cc:
                label_change[i][b] = color[b]
            else:
                color[b] = color_flip(color[b], n_class)
                label_change[i][b] = color[b]

    return label_change

def gen_episode_with_seg_reward(x, x_mask, y, sess, sample_graph, args):

    sg = sample_graph

    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])
    y = np.transpose(y, [1,0])

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0)
    max_seq_len = int(max(seq_lens))

    prev_state = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])
    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    reward_update_pos = [0]*n_batch
    sample_done = [] # indices of examples done processing

    prev_action_idx = [-1]*n_batch
    prev_action_pos = [-1]*n_batch

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])
    pred_idx = np.zeros([max_seq_len, n_batch], dtype=np.int32)
    actions = np.zeros([max_seq_len-1, n_batch, args.n_action])
    rewards = np.zeros([max_seq_len-1, n_batch])
    action_entropies = np.zeros([max_seq_len-1, n_batch])

    # for recording
    full_action_samples = np.zeros([max_seq_len, n_batch, args.n_action])
    full_action_probs = np.zeros([max_seq_len, n_batch, args.n_action])

    # for each time step (index j)
    for j, (x_step, y_step) in enumerate(itertools.izip(x, y)):
        _x_step, _y_step, _prev_state, target_indices = \
            filter_last(x_step, y_step, prev_state, j, seq_lens, sample_done)

        if len(_x_step):

            feed_dict={sg.step_x_data: _x_step}
            feed_prev_state(feed_dict, sg.init_state, _prev_state)

            step_label_likelihood_j, new_prev_state, step_pred_idx = \
                sess.run([sg.step_label_probs, sg.step_last_state, sg.step_pred_idx],
                    feed_dict=feed_dict)

            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(pred_idx, step_pred_idx, target_indices, update_pos)

            reward_target_indices = fill_seg_match_reward(rewards, y, j, 
                prev_action_idx, prev_action_pos, target_indices, reward_update_pos, update_pos, args)

            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)
            sample_done.extend(target_indices)

        _x_step, _y_step, _prev_state, target_indices = \
            filter_action_end(x_step, y_step, prev_state, j, action_counters, sample_done)

        if len(_x_step):
        
            feed_dict={sg.step_x_data: _x_step}
            feed_prev_state(feed_dict, sg.init_state, _prev_state)
           
            action_idx, step_action_prob_j, step_label_likelihood_j, new_prev_state, action_entropy, step_pred_idx = \
                sess.run([sg.step_action_samples, sg.step_action_probs, sg.step_label_probs,
                    sg.step_last_state, sg.action_entropy, sg.step_pred_idx],
                    feed_dict=feed_dict)
            
            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])
            action_one_hot = np.eye(args.n_action)[action_idx.flatten()]

            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(pred_idx, step_pred_idx, target_indices, update_pos)
            fill(action_entropies, action_entropy, target_indices, update_pos)
            fill(actions, action_one_hot, target_indices, update_pos)

            update_action_counters(action_counters, action_idx.flatten(), target_indices, args)

            reward_target_indices = fill_seg_match_reward(rewards, y, j,
                prev_action_idx, prev_action_pos, target_indices, reward_update_pos, update_pos, args)

            advance_pos(update_pos, target_indices)
            advance_pos(reward_update_pos, reward_target_indices)
            update_prev_state(prev_state, new_prev_state, target_indices)

            for label, idx in zip(step_pred_idx, target_indices):
                prev_action_idx[idx] = label
                prev_action_pos[idx] = j

            for i, s_idx in enumerate(target_indices):
                full_action_samples[j, s_idx] = action_one_hot[i]
                full_action_probs[j, s_idx] = step_action_prob_j[i]
        else:
            update_action_counters(action_counters, [], [], args)

    max_seq_len, mask, max_reward_seq_len, reward_mask = gen_mask2(update_pos, reward_update_pos, n_batch)

    outp = transpose_all([new_x[:max_seq_len],
                         new_y[:max_seq_len],                         
                         actions[:max_seq_len-1],
                         rewards[:max_reward_seq_len],
                         action_entropies[:max_seq_len-1],
                         mask,
                         reward_mask])

    output_image = gen_output_image(full_action_samples, y, args.n_class)
    outp.append(output_image)

    outp.extend(transpose_all([pred_idx[:max_seq_len]]))

    return outp

def get_seg_len(ref_labels):
    start_label = ref_labels[0]
    seg_len = 0
    for l in ref_labels:
        if l == start_label: seg_len += 1
        else:
            break

    return seg_len

def get_best_actions(target_indices, j, seq_lens, args, y):
    best_actions = []

    for idx in target_indices:
        max_jump = args.n_fast_action if args.n_fast_action > 0 else args.n_action
        upto = int(min(j+max_jump, seq_lens[idx]))
        ref_labels = y[j:upto, idx]
        seg_len = get_seg_len(ref_labels)
        best_actions.append(seg_len - 1)

    return np.asarray(best_actions)

def gen_supervision(x, x_mask, y, args):
    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])
    y = np.transpose(y, [1,0])

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0)
    max_seq_len = int(max(seq_lens))

    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    sample_done = [] # indices of examples done processing

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])
    actions = np.zeros([max_seq_len-1, n_batch])
    actions_1hot = np.zeros([max_seq_len-1, n_batch, args.n_action])

    full_action_samples = np.zeros([max_seq_len, n_batch, args.n_action])

    # for each time step (index j)
    for j, (x_step, y_step) in enumerate(itertools.izip(x, y)):
        _x_step, _y_step, target_indices = filter_last2(x_step, y_step, j, seq_lens, sample_done)

        if len(_x_step):
            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)

            advance_pos(update_pos, target_indices)
            sample_done.extend(target_indices)

        _x_step, _y_step, target_indices = filter_action_end2(x_step, y_step, j, action_counters, sample_done)

        if len(_x_step):
            best_actions = get_best_actions(target_indices, j, seq_lens, args, y)

            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(actions, best_actions, target_indices, update_pos)
            action_one_hot = np.eye(args.n_action)[best_actions]
            fill(actions_1hot, action_one_hot, target_indices, update_pos)
            
            update_action_counters(action_counters, best_actions, target_indices, args)

            advance_pos(update_pos, target_indices)
                
            # For visualization
            for i, s_idx in enumerate(target_indices):
                full_action_samples[j, s_idx] = action_one_hot[i]
        
        else:
            update_action_counters(action_counters, [], [], args)

    max_seq_len, mask = gen_mask3(update_pos, n_batch)
    outp = transpose_all([new_x[:max_seq_len],
                         new_y[:max_seq_len],                         
                         actions[:max_seq_len-1],
                         actions_1hot[:max_seq_len-1],
                         mask])


    output_image = gen_output_image(full_action_samples, y, args.n_class)
    outp.append(output_image)
    return outp

def skip_rnn_forward_parallel(x, x_mask, sess, sample_graph, n_fast_action):
    sg = sample_graph

    n_class = sample_graph.step_label_probs.shape[-1].value
    n_hidden = sample_graph.step_last_state.shape[-1].value
    n_action = sample_graph.step_action_probs.shape[-1].value

    # x shape is [time_step, batch_size, features]

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0)
    max_seq_len = int(max(seq_lens))

    # shape should be (2, n_batch, n_hidden) when it is used
    prev_state = np.zeros((n_batch, 2, n_hidden))

    # init counter and positions
    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    label_probs = np.zeros([max_seq_len, n_batch, n_class])
    actions_1hot = np.zeros([max_seq_len-1, n_batch, n_action]) 

    # for each step (index j)
    for j, x_step in enumerate(x):
        _x_step, _prev_state, target_indices = \
            filter_last_forward(x_step, prev_state, j, seq_lens, sample_done)

        # If final step sample exists,
        if len(_x_step):
            step_label_likelihood_j, new_prev_state = \
                sess.run([sg.step_label_probs, sg.step_last_state], 
                    feed_dict={sg.step_x_data: _x_step, sg.prev_states: np.transpose(_prev_state, [1, 0, 2])})
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)
            update_prev_state(prev_state, new_prev_state, target_indices)
         
            advance_pos(update_pos, target_indices)
            sample_done.extend(target_indices)

        _x_step, _prev_state, target_indices = \
            filter_action_end_forward(x_step, prev_state, j, action_counters, sample_done)

        # If sample exist, process
        if len(_x_step):
            action_idx, step_label_likelihood_j, new_prev_state = \
                sess.run([sg.step_action_samples, sg.step_label_probs, sg.step_last_state],
                    feed_dict={sg.step_x_data: _x_step, sg.prev_states: np.transpose(_prev_state, [1, 0, 2])})
            new_prev_state = np.transpose(new_prev_state, [1, 0, 2])
            
            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)

            action_one_hot = np.eye(n_action)[action_idx.flatten()]
            fill(actions_1hot, action_one_hot, target_indices, update_pos)

            update_prev_state(prev_state, new_prev_state, target_indices)

            update_action_counters2(action_counters, action_idx.flatten(), 
                target_indices, n_action, n_fast_action)

            advance_pos(update_pos, target_indices)
        else:
            update_action_counters2(action_counters, [], [], n_action, n_fast_action)

    new_max_seq_len, mask = gen_mask_from(update_pos)
    return transpose_all([actions_1hot[:new_max_seq_len-1], label_probs[:new_max_seq_len], mask])

def fixed_skip_forward(x, x_mask, sess, test_graph):
    tg = test_graph

    # n_batch, n_seq, n_feat -> n_seq, n_batch, n_feat
    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])

    n_class = tg.step_label_probs.shape[-1].value
    n_hidden = tg.step_last_state[0].c.shape[-1].value
    n_layer = len(tg.step_last_state)
    
    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0, dtype=np.int32)
    max_seq_len = max(seq_lens)

    prev_state = np.zeros([n_batch, n_layer, 2, n_hidden])
    target_indices = list(range(n_batch)) # always update all prev_state

    update_pos = [0]*n_batch
    label_probs = np.zeros([max_seq_len, n_batch, n_class])

    # for each step (index j)
    for j, x_step in enumerate(x):
        feed_dict={tg.step_x_data: x_step}
        feed_prev_state(feed_dict, tg.init_state, prev_state)

        step_label_likelihood_j, new_prev_state = \
            sess.run([tg.step_label_probs, tg.step_last_state], feed_dict=feed_dict)

        new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])
        fill(label_probs, step_label_likelihood_j, target_indices, update_pos)
       
        update_prev_state(prev_state, new_prev_state, target_indices)
        advance_pos(update_pos, target_indices)

    return transpose_all([label_probs[:max_seq_len]])

def skip_rnn_forward_parallel2(x, x_mask, sess, sample_graph, n_fast_action, no_sampling=False):
    sg = sample_graph

    # n_batch, n_seq, n_feat -> n_seq, n_batch, n_feat
    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])

    n_class = sg.step_label_probs.shape[-1].value
    n_hidden = sg.step_last_state[0].c.shape[-1].value
    n_action = sg.step_action_probs.shape[-1].value
    n_layer = len(sg.step_last_state)

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0, dtype=np.int32)
    max_seq_len = max(seq_lens)

    prev_state = np.zeros([n_batch, n_layer, 2, n_hidden])

    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    label_probs = np.zeros([max_seq_len, n_batch, n_class])
    actions_1hot = np.zeros([max_seq_len-1, n_batch, n_action])
    
    # for each step (index j)
    for j, x_step in enumerate(x):
        _x_step, _prev_state, target_indices = \
            filter_last_forward(x_step, prev_state, j, seq_lens, sample_done)

        if len(_x_step):
            feed_dict={sg.step_x_data: _x_step}
            feed_prev_state(feed_dict, sg.init_state, _prev_state)

            step_label_likelihood_j, new_prev_state = \
                sess.run([sg.step_label_probs, sg.step_last_state], feed_dict=feed_dict)

            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)
            update_prev_state(prev_state, new_prev_state, target_indices)
         
            advance_pos(update_pos, target_indices)
            sample_done.extend(target_indices)

        _x_step, _prev_state, target_indices = \
            filter_action_end_forward(x_step, prev_state, j, action_counters, sample_done)

        if len(_x_step):
            feed_dict={sg.step_x_data: _x_step}
            feed_prev_state(feed_dict, sg.init_state, _prev_state)

            if not no_sampling:
                action_idx, step_label_likelihood_j, new_prev_state = \
                    sess.run([sg.step_action_samples, sg.step_label_probs, sg.step_last_state],
                        feed_dict=feed_dict)
            else:
                step_action_probs, step_label_likelihood_j, new_prev_state = \
                sess.run([sg.step_action_probs, sg.step_label_probs, sg.step_last_state],
                    feed_dict=feed_dict)
                action_idx = np.argmax(step_action_probs, axis=-1)
            
            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)

            action_one_hot = np.eye(n_action)[action_idx.flatten()]
            fill(actions_1hot, action_one_hot, target_indices, update_pos)

            update_prev_state(prev_state, new_prev_state, target_indices)

            update_action_counters2(action_counters, action_idx.flatten(), 
                target_indices, n_action, n_fast_action)

            advance_pos(update_pos, target_indices)
        else:
            update_action_counters2(action_counters, [], [], n_action, n_fast_action)

    new_max_seq_len, mask = gen_mask_from(update_pos)
    return transpose_all([actions_1hot[:new_max_seq_len-1], label_probs[:new_max_seq_len], mask])


def get_actions_image(actions_taken):
    # n_batch, n_action, n_seq
    actions_img = np.transpose(actions_taken, [1, 2, 0])
    # n_batch, n_action, n_seq, 1
    actions_img = np.expand_dims(actions_img, axis=-1)
    # make it thicker: n_batch, n_action*5, n_seq*5, 1
    actions_img = np.repeat(actions_img, repeats=5, axis=1) 
    actions_img = np.repeat(actions_img, repeats=5, axis=2)

    return actions_img

def get_labels_image(y, n_class):

    # n_seq, n_batch
    y = to_label_change(y, n_class)

    # n_seq, n_batch, 1
    full_label_data = np.expand_dims(y, axis=-1)

    # n_batch, 1, n_seq
    full_label_data = np.transpose(full_label_data, [1, 2, 0])
    # n_batch, 1, n_seq, 1
    full_label_data = np.expand_dims(full_label_data, axis=-1)
    # make it thicker: n_batch, 1*5, n_seq*5, 1
    full_label_data = np.repeat(full_label_data, repeats=5, axis=1) 
    full_label_data = np.repeat(full_label_data, repeats=5, axis=2).astype(np.float32)
    full_label_data /= float(n_class)
    
    return full_label_data

def get_frame_reads_image(actions_taken):
    # n_batch, n_action, n_seq
    frame_reads = np.transpose(actions_taken, [1,2,0])
    # n_batch, 1, n_seq
    frame_reads = np.sum(frame_reads, axis=1, keepdims=True)
    # n_batch, 1, n_seq, 1
    frame_reads = np.expand_dims(frame_reads, axis=-1)
    # make it thicker: n_batch, 1*5, n_seq*5, 1
    frame_reads = np.repeat(frame_reads, repeats=5, axis=1) 
    frame_reads = np.repeat(frame_reads, repeats=5, axis=2)

    return frame_reads

def gen_output_image(actions_taken, y, n_class, pred_idx=None):
    # actions_taken: n_seq, n_batch, n_action
    # y: n_seq, n_batch
    # pred_idx: n_seq, n_batch

    actions_img = get_actions_image(actions_taken)
    labels_img = get_labels_image(y, n_class)    
    frame_reads = get_frame_reads_image(actions_taken)

    output_image = np.concatenate([
        # R channel (red)
        np.concatenate([labels_img, np.zeros_like(labels_img), np.zeros_like(labels_img)], axis=-1),
        # G channel (green)
        np.concatenate([np.zeros_like(frame_reads), frame_reads, np.zeros_like(frame_reads)], axis=-1),
        # B channel (blue)
        np.concatenate([np.zeros_like(actions_img), np.zeros_like(actions_img), actions_img], axis=-1)
    ], axis=1)

    return output_image

def gen_output_image_subsample(x, y, n_skip, start_idx):
    # n_seq, n_batch, n_action
    n_action = n_skip+1
    n_batch, n_seq, n_feat = x.shape
    actions_taken = np.zeros([n_batch, n_seq, n_action])

    action_idx = [n_action - 1 for i in range(start_idx, n_seq, n_action)]
    action_one_hot = np.eye(n_action)[action_idx]
    actions_taken[:,start_idx::n_action] = action_one_hot

    return gen_output_image(
        np.transpose(actions_taken, [1, 0, 2]), np.transpose(y, [1, 0]), n_action)

def choose(epsilon, a, b):
    if random.random() < epsilon:
        return a
    else:
        return b

def choose_actions(best_actions, pred_actions, args):
    new_actions = []
    for b, p in zip(best_actions, pred_actions):
        new_actions.append(choose(args.epsilon, b, p))
    return np.asarray(new_actions)

def get_random_pred_actions(target_indices, args):
    random_pred_actions = []

    for i in target_indices:
        random_pred_actions.append(random.randint(0, args.n_action-1))

    return np.asarray(random_pred_actions)

def gen_supervision_scheduled_sampling(x, y, x_mask, sess, test_graph, args):

    # n_batch, n_seq, n_feat -> n_seq, n_batch, n_feat
    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])
    y = np.transpose(y, [1,0])

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0, dtype=np.int32)
    max_seq_len = max(seq_lens)

    prev_state = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])

    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])

    actions_1hot = np.zeros([max_seq_len, n_batch, args.n_action])

    # for visualization
    actions_taken = np.zeros([max_seq_len, n_batch, args.n_action])

    # for each step (index j)
    for j, (x_step, y_step) in enumerate(itertools.izip(x,y)):
        _x_step, _y_step, _prev_state, target_indices = \
            filter_last(x_step, y_step, prev_state, j, seq_lens, sample_done)

        if len(_x_step):
            feed_dict={test_graph.step_x_data: _x_step}
            feed_prev_state(feed_dict, test_graph.init_state, _prev_state)

            step_label_likelihood_j, new_prev_state = \
                sess.run([test_graph.step_label_probs, test_graph.step_last_state], feed_dict=feed_dict)

            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            update_prev_state(prev_state, new_prev_state, target_indices)
         
            advance_pos(update_pos, target_indices)
            sample_done.extend(target_indices)

        _x_step, _y_step, _prev_state, target_indices = \
            filter_action_end(x_step, y_step, prev_state, j, action_counters, sample_done)

        if len(_x_step):
            feed_dict={test_graph.step_x_data: _x_step}
            feed_prev_state(feed_dict, test_graph.init_state, _prev_state)

            pred_actions, step_label_likelihood_j, new_prev_state = \
                sess.run([test_graph.step_action_samples, test_graph.step_label_probs, test_graph.step_last_state],
                    feed_dict=feed_dict)

            pred_actions = pred_actions.flatten() # output from tf.multinomial contains one more dimension
            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])
            best_actions = get_best_actions(target_indices, j, seq_lens, args, y)
            
            if args.use_random_prediction:
                pred_actions = get_random_pred_actions(target_indices, args)
            
            actions_chosen = choose_actions(best_actions, pred_actions, args)            
            supervision_action_1hot = np.eye(args.n_action)[best_actions.flatten()]
            real_action_1hot = np.eye(args.n_action)[actions_chosen.flatten()]

            fill(new_x, _x_step, target_indices, update_pos)
            fill(new_y, _y_step, target_indices, update_pos)
            fill(actions_1hot, supervision_action_1hot, target_indices, update_pos)

            # visualization
            for i, s_idx in enumerate(target_indices):
                actions_taken[j, s_idx] = real_action_1hot[i]

            update_prev_state(prev_state, new_prev_state, target_indices)

            update_action_counters2(action_counters, actions_chosen.flatten(), 
                target_indices, args.n_action, args.n_fast_action)

            advance_pos(update_pos, target_indices)
        else:
            update_action_counters2(action_counters, [], [], args.n_action, args.n_fast_action)

    new_max_seq_len, mask = gen_mask_from(update_pos)
    
    outp = transpose_all([new_x[:new_max_seq_len], new_y[:new_max_seq_len], actions_1hot[:new_max_seq_len-1], mask])
    output_image = gen_output_image(actions_taken, y, args.n_class)
    outp += [output_image,]
    
    return outp

def skip_rnn_forward_supervised(x, x_mask, sess, test_graph, n_fast_action, y=None):
    # y for visualization

    # n_batch, n_seq, n_feat -> n_seq, n_batch, n_feat
    x = np.transpose(x, [1,0,2])
    x_mask = np.transpose(x_mask, [1,0])

    n_class = test_graph.step_label_probs.shape[-1].value
    n_hidden = test_graph.step_last_state[0].c.shape[-1].value
    n_action = test_graph.step_action_probs.shape[-1].value
    n_layer = len(test_graph.step_last_state)

    n_seq, n_batch, n_feat = x.shape
    seq_lens = x_mask.sum(axis=0, dtype=np.int32)
    max_seq_len = max(seq_lens)

    prev_state = np.zeros([n_batch, n_layer, 2, n_hidden])

    action_counters = [0]*n_batch
    update_pos = [0]*n_batch
    sample_done = [] # indices of examples fully processed

    new_x = np.zeros([max_seq_len, n_batch, n_feat])
    new_y = np.zeros([max_seq_len, n_batch])
    label_probs = np.zeros([max_seq_len, n_batch, n_class])

    actions_1hot = np.zeros([max_seq_len, n_batch, n_action])

    # for visualization
    actions_taken = np.zeros([max_seq_len, n_batch, n_action])

    # for each step (index j)
    for j, x_step in enumerate(x):
        _x_step, _prev_state, target_indices = \
            filter_last_forward(x_step, prev_state, j, seq_lens, sample_done)

        if len(_x_step):
            feed_dict={test_graph.step_x_data: _x_step}
            feed_prev_state(feed_dict, test_graph.init_state, _prev_state)

            step_label_likelihood_j, new_prev_state = \
                sess.run([test_graph.step_label_probs, test_graph.step_last_state], feed_dict=feed_dict)

            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])

            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)
            update_prev_state(prev_state, new_prev_state, target_indices)
         
            advance_pos(update_pos, target_indices)
            sample_done.extend(target_indices)

        _x_step, _prev_state, target_indices = \
            filter_action_end_forward(x_step, prev_state, j, action_counters, sample_done)

        if len(_x_step):
            feed_dict={test_graph.step_x_data: _x_step}
            feed_prev_state(feed_dict, test_graph.init_state, _prev_state)

            action_idx, step_label_likelihood_j, new_prev_state = \
                sess.run([test_graph.step_action_samples, test_graph.step_label_probs, test_graph.step_last_state],
                    feed_dict=feed_dict)

            new_prev_state = np.transpose(np.asarray(new_prev_state), [2,0,1,3])
            action_1hot = np.eye(n_action)[action_idx.flatten()]

            fill(new_x, _x_step, target_indices, update_pos)
            fill(label_probs, step_label_likelihood_j, target_indices, update_pos)
            fill(actions_1hot, action_1hot, target_indices, update_pos)

            # visualization
            for i, s_idx in enumerate(target_indices):
                actions_taken[j, s_idx] = action_1hot[i]

            update_prev_state(prev_state, new_prev_state, target_indices)

            update_action_counters2(action_counters, action_idx.flatten(), 
                target_indices, n_action, n_fast_action)

            advance_pos(update_pos, target_indices)
        else:
            update_action_counters2(action_counters, [], [], n_action, n_fast_action)

    new_max_seq_len, mask = gen_mask_from(update_pos)
    
    outp = transpose_all([actions_1hot[:new_max_seq_len-1], 
        label_probs[:new_max_seq_len], mask])

    if y is not None: # visualization request
        y = np.transpose(y, [1,0])
        output_image = gen_output_image(actions_taken, y, n_class)
        outp += [output_image,]
    
    return outp

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

def compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args, final_cost=False):
    # shape: n_batch, n_seq, n_feat or n_batch, n_seq

    reward_mask_1d = new_reward_mask.reshape([-1])
    rewards_1d = rewards.reshape([-1])[reward_mask_1d==1.]
    discounted_rewards = []
    for reward, mask in zip(rewards, new_reward_mask):
        this_len = int(mask.sum())
        discounted_reward = discount(reward[:this_len], args.discount_gamma)
        if final_cost:
            discounted_reward = np.ones_like(discounted_reward)*discounted_reward[0]
        discounted_rewards.append(discounted_reward)

    reshape_new_x = new_x.reshape([-1, new_x.shape[2]])
    baseline_1d = vf.predict(reshape_new_x)
    baseline_2d = baseline_1d.reshape([new_x.shape[0], -1]) * new_x_mask

    advantages = np.zeros_like(rewards)
    discounted_rewards_arr = np.zeros_like(rewards)
    for i, (delta, mask) in enumerate(zip(baseline_2d, new_reward_mask)):
        this_len = int(mask.sum())
        advantages[i, :this_len] = discounted_rewards[i] - delta[:this_len]
        discounted_rewards_arr[i, :this_len] = discounted_rewards[i]

    advantages_1d = advantages.reshape([-1])[reward_mask_1d==1.]
    advantages = ((advantages - advantages_1d.mean()) / (advantages_1d.std()+1e-8)) * new_reward_mask

    valid_x_indices= np.where(new_reward_mask==1.)
    valid_new_x = new_x[valid_x_indices]
    discounted_rewards_1d = np.concatenate(discounted_rewards, axis=0)
    vf.fit(valid_new_x, discounted_rewards_1d)

    return advantages, discounted_rewards_arr


def _compute_discounted_rewards(rewards, new_reward_mask, discount_gamma):
    discounted_rewards = [] # Rt [n_batch, n_seq]

    for reward, mask in zip(rewards, new_reward_mask):
        # for each sample
        this_len = int(mask.sum())
        discounted_reward = discount(reward[:this_len], discount_gamma)
        discounted_rewards.append(discounted_reward)
    
    return discounted_rewards

def _compute_advantages(new_x, new_x_mask, vf, rewards, discounted_rewards, new_reward_mask):
    reshape_new_x = new_x.reshape([-1, new_x.shape[-1]]) # [n_batch * n_seq, n_hidden]
    baseline_1d = vf.predict(reshape_new_x) # [n_batch * n_seq]
    baseline_2d = baseline_1d.reshape([new_x.shape[0], -1]) * new_x_mask # [n_batch, n_seq]
    advantages = np.zeros_like(rewards)
    for i, (delta, mask) in enumerate(zip(baseline_2d, new_reward_mask)):
        this_len = int(mask.sum())
        advantages[i, :this_len] = discounted_rewards[i] - delta[:this_len]

    return advantages

def _normalize_advantages(advantages, new_reward_mask):
    reward_mask_1d = new_reward_mask.reshape([-1]) # n_batch * n_seq

    advantages_1d = advantages.reshape([-1])[reward_mask_1d==1.] # [n_batch * n_seq]
    advantages = ((advantages - advantages_1d.mean()) / (advantages_1d.std()+1e-8)) \
        * new_reward_mask

    return advantages

def compute_advantage2(new_x, new_x_mask, rewards, new_reward_mask, vf, args):
    # shape: [n_batch, n_seq, n_hidden] or [n_batch, n_seq] 
    discounted_rewards = _compute_discounted_rewards(rewards, new_reward_mask, args.discount_gamma)
    advantages = _compute_advantages(new_x, new_x_mask, vf, rewards, discounted_rewards, new_reward_mask) # [n_batch, n_seq]
    advantages = _normalize_advantages(advantages, new_reward_mask)

    valid_h_indices= np.where(new_reward_mask==1.)
    valid_new_x = new_x[valid_h_indices]
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

def expand_label_probs(actions_1hot, orig_x_mask, label_probs):
    new_label_probs = np.zeros([orig_x_mask.shape[0], orig_x_mask.shape[1], label_probs.shape[-1]])

    skip_info = np.argmax(actions_1hot, axis=2) + 1 # number of repeats

    # for each example
    for i, (s, p) in enumerate(zip(skip_info, label_probs)):
        # for each step
        start_idx = 0
        for s_step, p_step in itertools.izip_longest(s, p, fillvalue=1):
            new_label_probs[i,start_idx:start_idx+s_step] = p_step
            start_idx += s_step

    return new_label_probs

def expand_output(actions_1hot, mask, new_mask, output, n_fast_action=0):
    shape = list(mask.shape)
    if len(output.shape) > 2:
        shape.append(output.shape[-1])

    new_output = np.zeros(shape)
    n_action = actions_1hot.shape[-1]
    skip_info = np.argmax(actions_1hot, axis=2) + 1 # number of repeats
    new_seq_lens = new_mask.sum(axis=1, dtype=np.int32)
    seq_lens = mask.sum(axis=1, dtype=np.int32)

    # for each example
    for i, (s, p, new_slen, slen) in enumerate(zip(skip_info, output, new_seq_lens, seq_lens)):
        # for each step
        start_idx = 0

        for s_step, p_step in itertools.izip(s[:new_slen-1], p[:new_slen-1]):
            if n_fast_action > 0 and s_step == n_action: 
                s_step = n_fast_action

            end_idx = min(start_idx+s_step, slen-1)
            new_output[i,start_idx:end_idx] = p_step
            start_idx += s_step
        
        new_output[i,slen-1] = p[new_slen-1]

    return new_output


def interpolate_feat(input_data, num_skips, axis=1, use_bidir=True):
    # Fill skipped ones by repeating (forward)
    full_fwd_data = np.repeat(input_data, repeats=num_skips, axis=axis)

    if use_bidir:
        full_bwd_data = np.repeat(np.flip(input_data, axis=axis), repeats=num_skips, axis=axis)
        full_bwd_data = np.flip(full_bwd_data, axis=axis)

        return (full_fwd_data + full_bwd_data)*0.5
    else:
        return full_fwd_data







