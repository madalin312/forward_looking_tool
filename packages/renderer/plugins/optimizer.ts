import fs from 'node:fs'
import path from 'node:path'
import { createRequire, builtinModules } from 'node:module'
import type { Alias, Plugin, UserConfig } from 'vite'
import esbuild from 'esbuild'
import libEsm from 'lib-esm'
import { builtins } from './build-config'

export type DepOptimizationConfig = {
  include?: (string | {
    name: string
    /**
     * Explicitly specify the module type
     * - `commonjs` - Only the ESM code snippet is wrapped
     * - `module` - First build the code as cjs via esbuild, then wrap the ESM code snippet
     */
    type?: "commonjs" | "module"
  })[]
  buildOptions?: import('esbuild').BuildOptions
  // TODO: consider support webpack 🤔
  // webpack?: import('webpack').Configuration
}

const cjs_require = createRequire(import.meta.url)
const CACHE_DIR = '.vite-electron-renderer'
const name = 'vite-plugin-electron-renderer:optimizer'

let root: string
let node_modules_path: string

export default function optimizer(options: DepOptimizationConfig = {}): Plugin[] | undefined {
  const { include, buildOptions } = options

  return [
    {
      // Built-in modules should always be Pre-Bundling.
      name: `${name}:built-in`,
      config(config) {
        root = config.root ? path.resolve(config.root) : process.cwd()
        node_modules_path = node_modules(root)

        fs.rmSync(path.join(node_modules_path, CACHE_DIR), { recursive: true, force: true })

        const aliases: Alias[] = [
          {
            find: 'electron',
            replacement: 'vite-plugin-electron-renderer/electron-renderer',
          },
          ...builtins
            .filter(m => m !== 'electron')
            .filter(m => !m.startsWith('node:'))
            .map<Alias>(m => ({
              find: new RegExp(`^(node:)?${m}$`),
              replacement: `vite-plugin-electron-renderer/builtins/${m}`,
            })),
        ]

        modifyAlias(config, aliases)
        modifyOptimizeDeps(config, builtins.concat(aliases.map(({ replacement }) => replacement)))
      },
    },
    {
      name: `${name}:npm-pkgs`,
      // At `vite build` phase, Node.js npm-pkgs can be built correctly by Vite.
      // TODO: consider support `vite build` phase, like Vite v3.0.0
      apply: 'serve',
      async config(config) {
        if (!include?.length) return

        const deps: {
          esm?: string
          cjs?: string
          filename?: string
        }[] = []
        const aliases: Alias[] = []
        const optimizeDepsExclude = []

        for (const item of include) {
          let name: string
          let type: string | undefined
          if (typeof item === 'string') {
            name = item
          } else {
            name = item.name
            type = item.type
          }
          if (type === 'module') {
            deps.push({ esm: name })
            continue
          }
          if (type === 'commonjs') {
            deps.push({ cjs: name })
            continue
          }
          if (builtins.includes(name)) {
            // Process in `built-in` plugin
            continue
          }

          const pkgId = path.join(node_modules_path, name, 'package.json')
          if (fs.existsSync(pkgId)) {
            // bare module
            const pkg = cjs_require(pkgId)
            if (pkg.type === 'module') {
              deps.push({ esm: name })
              continue
            }
            deps.push({ cjs: name })
            continue
          }

          const tmp = path.join(node_modules_path, name)
          try {
            // dirname or filename 🤔
            // `foo/bar` or `foo/bar/index.js`
            const filename = cjs_require.resolve(tmp)
            if (path.extname(filename) === '.mjs') {
              deps.push({ esm: name, filename })
              continue
            }
            deps.push({ cjs: name, filename })
            continue
          } catch (error) {
            console.log('Can not resolve path:', tmp)
          }
        }

        for (const dep of deps) {
          if (!dep.filename) {
            const module = (dep.cjs || dep.esm) as string
            try {
              dep.filename = cjs_require.resolve(module)
            } catch (error) {
              console.log('Can not resolve module:', module)
            }
          }
          if (!dep.filename) {
            continue
          }

          if (dep.cjs) {
            cjsBundling({
              name: dep.cjs,
              require: dep.cjs,
              requireId: dep.filename,
            })
          } else if (dep.esm) {
            esmBundling({
              name: dep.esm,
              entry: dep.filename,
              buildOptions,
            })
          }

          const name = dep.cjs || dep.esm
          if (name) {
            optimizeDepsExclude.push(name)
            const { destname } = dest(name)
            aliases.push({ find: name, replacement: destname })
          }
        }

        modifyAlias(config, aliases)
        modifyOptimizeDeps(config, optimizeDepsExclude)
      },
    },
  ]
}

function cjsBundling(args: {
  name: string
  require: string
  requireId: string
}) {
  const { name, require, requireId } = args
  const { exports } = libEsm({ exports: Object.keys(cjs_require(requireId)) })
  const code = `const _M_ = require("${require}");\n${exports}`
  writeFile({ name, code })
}

async function esmBundling(args: {
  name: string,
  entry: string,
  buildOptions?: esbuild.BuildOptions,
}) {
  const { name, entry, buildOptions } = args
  const { name_cjs, destname_cjs } = dest(name)
  return esbuild.build({
    entryPoints: [entry],
    outfile: destname_cjs,
    target: 'node14',
    format: 'cjs',
    bundle: true,
    sourcemap: true,
    external: [
      ...builtinModules,
      ...builtinModules.map(mod => `node:${mod}`),
    ],
    ...buildOptions,
  }).then(result => {
    if (!result.errors.length) {
      cjsBundling({
        name,
        require: `${CACHE_DIR}/${name}/${name_cjs}`,
        requireId: destname_cjs,
      })
    }
    return result
  })
}

function writeFile(args: {
  name: string
  code: string
}) {
  const { name, code } = args
  const { destpath, destname } = dest(name)
  !fs.existsSync(destpath) && fs.mkdirSync(destpath, { recursive: true })
  fs.writeFileSync(destname, code)
  console.log('Pre-bundling:', name)
}

function dest(name: string) {
  const destpath = path.join(node_modules_path, CACHE_DIR, name)
  const name_js = 'index.js'
  const name_cjs = 'index.cjs'
  !fs.existsSync(destpath) && fs.mkdirSync(destpath, { recursive: true })
  return {
    destpath,
    name_js,
    name_cjs,
    destname: path.join(destpath, name_js),
    destname_cjs: path.join(destpath, name_cjs),
  }
}

function modifyOptimizeDeps(config: UserConfig, exclude: string[]) {
  config.optimizeDeps ??= {}
  config.optimizeDeps.exclude ??= []
  config.optimizeDeps.exclude.push(...exclude)
}

function modifyAlias(config: UserConfig, aliases: Alias[]) {
  config.resolve ??= {}
  config.resolve.alias ??= []
  if (Object.prototype.toString.call(config.resolve.alias) === '[object Object]') {
    config.resolve.alias = Object
      .entries(config.resolve.alias)
      .reduce<Alias[]>((memo, [find, replacement]) => memo.concat({ find, replacement }), [])
  }
  (config.resolve.alias as Alias[]).push(...aliases)
}

function node_modules(root: string, count = 0): string {
  if (node_modules.p) {
    return node_modules.p
  }
  const p = path.join(root, 'node_modules')
  if (fs.existsSync(p)) {
    return node_modules.p = p
  }
  if (count >= 19) {
    throw new Error('Can not found node_modules directory.')
  }
  return node_modules(path.join(root, '..'), count + 1)
}
// For ts-check
node_modules.p = ''
