// import * as tf from '@tensorflow/tfjs-core'

/**
 * Picks different number of values from each row
    in x depending on the values in counts.
 */
export function batchPackGraph (x, counts, numRows) {
  const outputs = []
  for (let i = 0; i < numRows; i++) {
    // outputs.push(x[i, :counts[i]]) // TODO
  }
  return tf.concat(outputs, 0)
}

/**
 * Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
 */

export function normBoxesGraph (boxes, shape) {
  const [h, w] = tf.split(tf.cast(shape, 'float32'), 2)
  const scale = tf.sub(tf.concat([h, w, h, w], -1), tf.tensor([1]))
  const shift = tf.tensor([0, 0, 1, 1])
  return tf.div(tf.sub(boxes, shift), scale)
}

/**
 * Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
 */

export function denormBoxesGraph (boxes, shape) {
  const [h, w] = tf.split(tf.cast(shape, tf.float32), 2)
  const scale = tf.concat([h, w, h, w], -1) - tf.constant(1)
  const shift = tf.constant([0, 0, 1, 1])
  return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
}
