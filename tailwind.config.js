module.exports = {
	content: [
		'./packages/renderer/src/**/*.{js,ts,jsx,tsx}',
		'node_modules/daisyui/dist/**/*.js'
	],
	plugins: [require('daisyui')]
}
