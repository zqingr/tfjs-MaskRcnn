// import * as tf from '@tensorflow/tfjs'
// import * as tf from '@tensorflow/tfjs-core'
import NormBoxesLayer from './customLayer/NormBoxesLayer'

class MaskRCNN {
  /**
   * @param {} mode Either "training" or "inference"
   * @param {*} config A Sub-class of the Config class
   * @param {*} modelDir Directory to save training logs and trained weights
   */
  constructor (mode, config, modelDir) {
    if (mode !== 'training' && mode !== 'inference') {
      throw new Error('Mode must be either training or inference')
    }

    this.mode = mode
    this.config = config
    this.modelDir = modelDir // TODO
    this.tfModel = this.build()
  }

  /**
   * Build Mask R-CNN architecture.
   *
   *  input_shape: The shape of the input image.
   *  mode: Either "training" or "inference". The inputs and
   *      outputs of the model differ accordingly.
   */
  build () {
    const [h, w] = this.config.IMAGE_SHAPE

    if (h / 2 ** 6 !== parseInt(h / 2 ** 6) || w / 2 ** 6 !== parseInt(w / 2 ** 6)) {
      throw new Error(`
          Image size must be dividable by 2 at least 6 times
          to avoid fractions when downscaling and upscaling.
          For example, use 256, 320, 384, 448, 512, ... etc. 
      `)
    }

    // Inputs
    const input_image = tf.input({ shape: [null, null, this.config.IMAGE_SHAPE[2]], name: 'input_image' })
    const input_image_meta = tf.input({ shape: [this.config.IMAGE_META_SIZE], name: 'input_image_meta' })

    if (this.mode === 'training') {
      // RPN GT
      const input_rpn_match = tf.input({ shape: [null, 1], name: 'input_rpn_match', dtype: 'int32' })
      const input_rpn_bbox = tf.input({ shape: [null, 4], name: 'input_rpn_bbox', dtype: 'float32' })

      // Detection GT (class IDs, bounding boxes, and masks)
      // 1. GT Class IDs (zero padded)
      const input_gt_class_ids = tf.input({ shape: [null, 1], name: 'input_gt_class_ids', dtype: 'int32' })
      // 2. GT Boxes in pixels (zero padded)
      // [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
      const input_gt_boxes = tf.input({ shape: [null, 4], name: 'input_gt_boxes', dtype: 'float32' })

      // Normalize coordinates
      const gt_boxes = new NormBoxesLayer(input_image.shape.slice(1, 3)).apply(input_gt_boxes)

      let input_gt_masks

      if (this.config.USE_MINI_MASK) {
        // 3. GT Masks  (zero padded)
        input_gt_masks = tf.input({ shape: [this.config.MINI_MASK_SHAPE[0], this.config.MINI_MASK_SHAPE[1], null], name: 'input_gt_masks', dtype: 'bool' })
      } else {
        input_gt_masks = tf.input({ shape: [h, w, null], name: 'input_gt_masks', dtype: 'bool' })
      }
      // GT Masks (zero padded)
      // [batch, height, width, MAX_GT_INSTANCES] of class IDs
    } else if (this.mode === 'inference') {
      // Anchors in normalized coordinates
      const input_anchors = tf.input({ shape: [null, 4], name: 'input_anchors' })
    }

    // Build the shared convolutional layers.
    // Bottom-up Layers
    // Returns a list of the last layers of each stage, 5 in total.
    // Don't create the thead (stage 5), so we pick the 4th item in the list.
    let backbone
    if (typeof this.config.BACKBONE === 'function') {
      backbone = this.config.BACKBONE(input_image, this.config)
    } else {
      backbone = resnet_graph(input_image, this.config.BACKBONE, true, this.config.TRAIN_BN)
    }

    const [_, C2, C3, C4, C5] = backbone
  }
}

export default MaskRCNN
