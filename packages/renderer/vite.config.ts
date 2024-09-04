import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import renderer from './plugins'
import pkg from '../../package.json'

// https://vitejs.dev/config/
export default defineConfig({
	root: __dirname,
	mode: process.env.NODE_ENV,
	base: './',
	plugins: [
		preact(),
		renderer({
			nodeIntegration: true
		})
	],
	build: {
		outDir: '../../dist/renderer',
		emptyOutDir: true,
		sourcemap: true
	},
	server: process.env.VSCODE_DEBUG
		? {
				host: pkg.debug.env.VITE_DEV_SERVER_HOSTNAME,
				port: pkg.debug.env.VITE_DEV_SERVER_PORT
		  }
		: undefined
})
