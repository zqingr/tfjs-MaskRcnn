/**
 * The identity_block is the block that has no conv layer at shortcut
 * Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the nb_filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      use_bias: Boolean. To use or not use a bias in conv layers.
      train_bn: Boolean. Train or freeze Batch Norm layers
 */

export function identity_block (input_tensor, kernel_size, filters, stage, block, use_bias, train_bn) {
  const [nb_filter1, nb_filter2, nb_filter3] = filters
  const conv_name_base = 'res' + stage + block + '_branch'
  const bn_name_base = 'bn' + stage + block + '_branch'

  // Main path
  let x = tf.layers
    .conv2d({ filters: nb_filter1, kernelSize: 1, useBias: use_bias, name: conv_name_base + '2a' })
    .apply(input_tensor)
  x = tf.layers
    .batchNormalization({ name: bn_name_base + '2a' })
    .apply(x, { training: train_bn })
  x = tf.layers.activation({ activation: 'relu' }).apply(x)

  x = tf.layers
    .conv2d({ filters: nb_filter2, kernelSize: kernel_size, useBias: use_bias, name: conv_name_base + '2b', padding: 'same' })
    .apply(x)
  x = tf.layers
    .batchNormalization({ name: bn_name_base + '2b' })
    .apply(x, { training: train_bn })
  x = tf.layers.activation({ activation: 'relu' }).apply(x)

  x = tf.layers
    .conv2d({ filters: nb_filter3, kernelSize: 1, useBias: use_bias, name: conv_name_base + '2c' })
    .apply(x)
  x = tf.layers
    .batchNormalization({ name: bn_name_base + '2c' })
    .apply(x, { training: train_bn })

  // Add input tensor
  x = tf.layers.add({ name: 'res' + stage + block }).apply([x, input_tensor])
  x = tf.layers.activation({ activation: 'relu', name: 'res' + stage + block + '_out' }).apply(x)
  return x
}
