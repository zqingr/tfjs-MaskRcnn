/**
 * Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
 */
function parseImageMetaGraph (meta) {
  return {
    image_id: meta[0],
    original_image_shape: meta.slice(1, 4),
    image_shape: meta.slice(4, 7),
    window: meta.slice(7, 11),
    scale: meta[11],
    active_class_ids: meta.slice(12, meta.length)
  }
}

export default parseImageMetaGraph
