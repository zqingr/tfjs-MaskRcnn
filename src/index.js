// import '@tensorflow/tfjs-layers/dist/exports'
// import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-webgl'

import MaskRCNN from './model/index'
import Config from './config'

const a = tf.tensor([1, 2, 3, 4])
const b = tf.sum(a)

const config = new Config()
const model = new MaskRCNN('training', config)
console.log(222, config, model)
