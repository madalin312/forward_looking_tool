import { ChildProcessWithoutNullStreams, spawn } from 'child_process'
import path from 'path'
import { Hypothesis } from './Components/HypoModal'
import fs from 'fs/promises'

export const venvPath = path.join(
	process.env.APPDATA || '',
	'ForwardLookingTool',
	'venv'
)

interface Params {
	scriptsPath: string
	dirPath: string
	hypo: Hypothesis
	file: string
	modelVars: string[]
	independentVars: number
	log: (m: string) => void
	threads: number
	mode: string
	resizeSample: number
}

export default class Runner {
	params: Params
	installing = false

	constructor(params: Params) {
		this.params = params
	}

	process?: ChildProcessWithoutNullStreams

	main(portfolio: string, reportingDate: string) {
		this.installing = false
		this.process?.kill()

		const {
			scriptsPath,
			dirPath,
			hypo,
			file,
			modelVars,
			independentVars,
			log,
			threads,
			mode,
			resizeSample
		} = this.params

		let scriptFile = 'script.py'
		switch (mode) {
			case 'bt':
				scriptFile = 'script_bt.py'
				break
			case 'bbva':
				scriptFile = 'script_bbva.py'
				break
			default:
				break
		}

		return new Promise<void>((resolve, reject) => {
			this.process = spawn(path.join(venvPath, 'Scripts', 'python.exe'), [
				path.join(scriptsPath, scriptFile),
				JSON.stringify({
					dirPath,
					hypo,
					file,
					independentVars,
					portfolio,
					reportingDate,
					modelVars,
					threads,
					resizeSample
				})
			])

			log('Started script, please wait...')

			this.process.on('error', (e) => {
				log('! ' + e.toString())

				if (e.message.includes('ENOENT')) {
					this.setupVenv().then(() =>
						this.main(portfolio, reportingDate).then(() =>
							resolve()
						)
					)
				} else reject(e)
			})

			this.process.stdout.on('data', (e) => {
				log(e.toString())
			})
			this.process.stderr.on('data', (e) => {
				if (e.toString().includes('ModuleNotFoundError')) {
					this.pipInstall().then(() =>
						this.main(portfolio, reportingDate).then(() =>
							resolve()
						)
					)
				}
				if (!e.toString().toLowerCase().includes('warning'))
					log('! ' + e.toString())
			})
			this.process.on('close', async (e) => {
				if (this.installing) return
				if (this.process?.exitCode === 0) {
					const dir = path.join(dirPath, `${portfolio}_${hypo?.name}`)
					const files = await fs.readdir(
						path.join(dirPath, `${portfolio}_${hypo?.name}`)
					)
					const summaryFile = files.find((f) =>
						f.startsWith('executive_summary_model_')
					)
					if (summaryFile == null) {
						log('! Failed to find summary')
						resolve()
						return
					}

					const summary = (
						await fs.readFile(path.join(dir, summaryFile))
					)
						.toString()
						.trim()

					log(
						`Finished. There were ${
							summary.match(
								/The number of models generated is: (\d+)/
							)?.[1]
						} model candidates identified. {link}`
					)

					resolve()
				} else reject(e)
			})
		})
	}

	async rInstall() {
		this.process?.kill()
		const { log, scriptsPath } = this.params

		log('Making sure R packages are installed...')
		return new Promise<void>((resolve, reject) => {
			this.process = spawn('Rscript', [
				path.join(scriptsPath, 'installPackages.R')
			])
			this.process.stdout.on('data', (e) => {
				log(e.toString())
			})
			this.process.stderr.on('data', (e) => {
				log('! ' + e.toString())
			})
			this.process.on('close', (e) => {
				if (this.process?.exitCode === 0) resolve()
				else reject(e)
			})
			this.process.on('error', (e) => {
				log(
					'! ' +
						e.toString() +
						'\n' +
						((e.message.includes('ENOENT') &&
							'Please make sure R is correctly installed.') ||
							'')
				)
				reject(e)
			})
		})
	}

	setupVenv() {
		this.installing = true
		this.process?.kill()
		const { log } = this.params

		log('Setting up venv...')
		return new Promise<void>((resolve, reject) => {
			this.process = spawn('python', ['-m', 'venv', venvPath])
			this.process.stdout.on('data', (e) => {
				log(e.toString())
			})
			this.process.stderr.on('data', (e) => {
				log('! ' + e.toString())
			})
			this.process.on('close', (e) => {
				if (this.process?.exitCode === 0) resolve()
				else reject(e)
				this.installing = false
			})
			this.process.on('error', (e) => {
				log(
					'! ' +
						e.toString() +
						'\n' +
						((e.message.includes('ENOENT') &&
							'Please make sure Python is correctly installed.') ||
							'')
				)

				reject(e)
			})
		})
	}

	pipInstall() {
		this.installing = true
		this.process?.kill()
		const { scriptsPath, log } = this.params
		log(
			'! Python dependency not found. Installing requirements, please wait...'
		)
		return new Promise<void>((resolve, reject) => {
			this.process = spawn(path.join(venvPath, 'Scripts', 'pip.exe'), [
				'install',
				'-r',
				path.join(scriptsPath, 'requirements.txt')
			])
			this.process.stdout.on('data', (e) => {
				if (!e.toString().includes('Requirement already satisfied'))
					log(e.toString())
			})
			this.process.stderr.on('data', (e) => {
				log('! ' + e.toString())
			})
			this.process.on('close', (e) => {
				this.installing = false
				if (this.process?.exitCode === 0) resolve()
				else reject(e)
			})
		})
	}
}
