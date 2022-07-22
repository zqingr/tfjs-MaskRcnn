import * as tf from '@tensorflow/tfjs-core'
/**
 * Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
 */

function normBoxesGraph (boxes, shape) {
  const height = shape[0]
  const width = shape[1]
  const boxs = tf.tensor(boxes)
  const y1 = boxs.slice([0, 0], [-1, 1])
  const x1 = boxs.slice([0, 1], [-1, 1])
  const y2 = boxs.slice([0, 2], [-1, 1])
  const x2 = boxs.slice([0, 3], [-1, 1])
  const y1Norm = y1.div(height)
  const x1Norm = x1.div(width)
  const y2Norm = y2.div(height)
  const x2Norm = x2.div(width)
  const boxsNorm = tf.concat([y1Norm, x1Norm, y2Norm, x2Norm], 1)
  return boxsNorm
}

export default normBoxesGraph
