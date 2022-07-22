import { normBoxesGraph } from '../utils/miscellenousGraphFunctions'

export default class NormBoxesLayer extends tf.layers.Layer {
  constructor (shape) {
    super({})
    this.shape = shape
  }

  call (x) {
    return normBoxesGraph(x, this.shape)
  }

  static get className () {
    return 'NormBoxesLayer'
  }
}

// tf.serialization.registerClass(NormBoxesLayer)
