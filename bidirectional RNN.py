__author__ = 'zhilonglu'
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm]*number_layers)
state = stacked_lstm.zero_state(batch_size,tf.float32)

loss = 0.0
for i in range(num_steps):
    if i>0:
        tf.get_varible_scope().reuse_variables()
    stacked_lstm_output,state = stacked_lstm(current_input,state)
    final_output = fully_connected(stacked_lstm_ouput)
    loss += calc_loss(final_output,expected_output)