process.env.DIST = join(__dirname, '..')
process.env['ELECTRON_DISABLE_SECURITY_WARNINGS'] = 'true'

import os from 'os'
import { join } from 'path'
import { app, BrowserWindow, dialog } from 'electron'
import remote from '@electron/remote/main'

app.setPath('userData', join(process.env.APPDATA || '', 'ForwardLookingTool'))

remote.initialize()

const isWin7 = os.release().startsWith('6.1')
if (isWin7) app.disableHardwareAcceleration()

let win: BrowserWindow | null = null

async function createWindow() {
	win = new BrowserWindow({
		webPreferences: {
			contextIsolation: false,
			nodeIntegration: true
		},
		height: 580,
		width: 700,
		maxHeight: 580,
		maxWidth: 700,
		minHeight: 580,
		minWidth: 700,
		maximizable: false,
		autoHideMenuBar: true,
		title: 'Forward Looking Tool'
	})

	remote.enable(win.webContents)

	if (process.env.VITE_DEV_SERVER_URL) {
		win.loadURL(process.env.VITE_DEV_SERVER_URL)
	} else {
		win.loadFile(join(process.env.DIST, 'renderer/index.html'))
		// win.webContents.openDevTools({ mode: 'undocked' })
	}

	let prevent = true

	win.on('close', async (e) => {
		if (!prevent) return

		e.preventDefault()

		dialog
			.showMessageBox({
				title: 'Confirm Exit',
				message: 'Are you sure? This will cancel all running scripts.',
				buttons: ['Yes', 'Cancel'],
				type: 'warning'
			})
			.then((r) => {
				if (r.response == 0) {
					prevent = false
					win?.close()
				}
			})
	})
}

app.on('ready', createWindow)
app.on('activate', createWindow)

app.on('window-all-closed', () => {
	app.quit()
})
