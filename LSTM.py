__author__ = 'zhilonglu'
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)
state = lstm.zero_state(batch_size,tf.float32)

loss = 0.0
for i in range(num_steps):
    if i>0:
        tf.get_varible_scope().reuse_variables()
    lstem_output,state = lstm(current_input,state)
    final_output = fully_connected(lstm_ouput)
    loss += calc_loss(final_output,expected_output)
