import babel from 'rollup-plugin-babel'
// import postcss from 'rollup-plugin-postcss'
import progress from 'rollup-plugin-progress'
import filesize from 'rollup-plugin-filesize'
import resolve from 'rollup-plugin-node-resolve'
import commonjs from 'rollup-plugin-commonjs'

export default {
  context: 'window',
  output: {
    strict: false,
    file: 'dist/index.js',
    format: 'cjs'
  },
  plugins: [
    // postcss({
    //   extract: 'index.min.CSS',
    // }),
    progress(),
    filesize({
      showGzippedSize: false
    }),
    resolve({
      // preferBuiltins: false
    }),
    commonjs(),
    babel({
      exclude: 'node_modules/**'
    })
  ]
}
