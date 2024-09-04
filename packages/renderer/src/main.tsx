import { h, render } from 'preact'
import App from './App'
import './main.css'

render(<App />, document.querySelector('#app') as Element)

window.matchMedia('(prefers-color-scheme: dark)').matches &&
	document.body.parentElement?.setAttribute('data-theme', 'business')
