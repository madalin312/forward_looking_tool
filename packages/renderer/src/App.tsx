import { Component, Fragment } from 'preact'
import { h, createRef } from 'preact'
import Input from './Components/Input'
import DirectoryInput from './Components/DirectoryInput'
import path from 'path'
import LogModal, { secondsToTime } from './Components/LogModal'
import HypoModal, { Hypothesis, defaultHypos } from './Components/HypoModal'
import FileInput from './Components/FileInput'
import type XlsxType from 'xlsx'
const xlsx: typeof XlsxType = require('xlsx')
import './style.sass'
import fs from 'fs/promises'
import pkg from '../../../package.json'
import Runner from './RunPythonScript'
import CheckboxModal from './Components/CheckboxModal'
import { promisify } from 'util'
import FieldHelp from './Components/FieldHelp'
import os from 'node:os'
import logo from './logo.svg'
import DocModal from './Components/DocModal'

const delay = promisify(setTimeout)

interface Props {}
interface State {
	dirPath?: string
	log: string[]
	hypo?: Hypothesis
	file?: string
	portfolios: string[]
	xHeader?: string[]
	modelVars: string[]
	invalidInput?: boolean
	elapsed: number
	running?: boolean
	currentPortfolio?: string
	finished?: boolean
	resizeSample: number
	mode: string
}

export default class App extends Component<Props, State> {
	independentVars?: number
	formRef = createRef<HTMLFormElement>()
	rows?: Record<string, any>[]
	runner?: Runner
	threads = Math.round(os.cpus().length / 2)

	constructor(props: Props) {
		super(props)

		this.state = {
			modelVars: [],
			log: [],
			portfolios: [],
			hypo: defaultHypos[0],
			elapsed: 0,
			resizeSample: 0,
			mode: 'normal'
		}
	}

	componentDidUpdate(
		previousProps: Readonly<Props>,
		previousState: Readonly<State>,
		snapshot: any
	): void {
		if (previousState.mode != this.state.mode) {
			this.setState({
				hypo:
					defaultHypos.find((h) => h.mode == this.state.mode) ||
					defaultHypos[0]
			})
		}
	}

	calculatePortfolioDate(portfolio: string) {
		if (this.rows == null) throw new Error()

		const lastRow = this.rows
			.filter((r) => {
				// only leave rows that have the portfolio column
				for (const [k, v] of Object.entries(r)) {
					if (k.trim() == portfolio && v === null) return false
				}
				return true
			})
			.pop() // get last valid row

		if (lastRow == null) return

		const date = new Date('1900-01-01T00:00:00.000Z')

		date.setUTCDate((lastRow['Date'] || lastRow['date']) - 1)

		return date.toISOString().split('T')[0]
	}

	async runScript() {
		this.setState(
			{
				elapsed: -1,
				running: true,
				currentPortfolio: undefined,
				finished: false
			},
			() => {
				const incElapsed = () => {
					if (!this.state.running || this.state.finished) return
					this.setState({
						elapsed: this.state.elapsed + 1
					})
					setTimeout(() => incElapsed(), 1000)
				}

				incElapsed()
			}
		)

		if (!this.formRef.current?.reportValidity()) {
			setTimeout(() => {
				;(
					document.querySelector('#logModal') as HTMLInputElement
				).checked = false
				;(document.querySelector('#test') as HTMLButtonElement)?.click()
			}, 10)
			return
		}
		const {
			dirPath,
			hypo,
			file,
			portfolios,
			modelVars,
			mode,
			resizeSample
		} = this.state
		let { independentVars, threads } = this
		this.setState({
			log: []
		})

		independentVars = independentVars || modelVars.length + 1

		let isBuilt = true

		try {
			await fs.stat(path.join(process.cwd(), 'resources/app'))
		} catch (e) {
			isBuilt = false
		}

		const scriptsPath =
			(isBuilt &&
				path.join(process.cwd(), 'resources', 'app', 'python')) ||
			path.join(process.cwd(), 'python')

		if (
			hypo == null ||
			file == null ||
			portfolios == null ||
			dirPath == null ||
			independentVars == null
		)
			throw new Error()

		const log = (m: string) =>
			this.setState({
				log: [...this.state.log, m]
			})

		this.runner = new Runner({
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
		})
		
		await this.runner.rInstall()

		try {
			let ranFirstPortfolio = false
			for (const portfolio of portfolios) {
				if (ranFirstPortfolio) {
					log(`Running portfolio ${portfolio} in 10 seconds...`)
					// await delay(10 * 1000)
				}
				ranFirstPortfolio = true
				this.setState({
					currentPortfolio: portfolio,
					elapsed: 0
				})
				const date = this.calculatePortfolioDate(portfolio)
				if (date == null) throw new Error()
				await this.runner?.main(portfolio, date)

				log(
					`Took ${secondsToTime(
						this.state.elapsed
					)} to run portfolio ${portfolio}.`
				)
			}
		} catch (e) {
			console.trace(e)
		} finally {
			this.setState({
				finished: true
			})
		}
	}

	render() {
		const {
			dirPath,
			log,
			hypo,
			file,
			xHeader,
			modelVars,
			portfolios,
			elapsed,
			running,
			currentPortfolio,
			finished,
			resizeSample,
			mode
		} = this.state

		return (
			<Fragment>
				<code class="version text-base-content">v{pkg.version}</code>
				<img src={logo} class="logo" />
				<form ref={this.formRef}>
					<div class="form-control">
						<label class="label">
							<span class="label-text">Entity Type</span>
							<FieldHelp id="modehelp">
								<h1 class="font-bold text-xl">Entity Type</h1>
								<p>
									Choose the type of the entity. Currently
									implemented types are Normal, BBVA and BT (Banca
									Transilvania)
								</p>
							</FieldHelp>
						</label>
						<select
							onInput={(e) =>
								this.setState({
									mode: e.currentTarget.value
								})
							}
							class="select select-bordered w-full max-w-xs"
							value={mode}
							required>
							<option selected value="normal">
								Normal
							</option>
							<option value="bt">BT</option>
							<option value="bbva">BBVA</option>
							<option value="optimized">Optimized</option>
						</select>
					</div>
					{mode == 'bt' && (
						<div class="form-control">
							<label class="label">
								<span class="label-text">Resize Sample</span>
								<FieldHelp id="resizesamplehelp">
									<h1 class="font-bold text-xl">
										Resize Sample
									</h1>
									<p>
										This parameter is used to establish if
										linear interpolated data is used.
										Default is 0, meaning the first row is
										not considered in development
									</p>
								</FieldHelp>
							</label>
							<select
								onInput={(e) =>
									this.setState({
										resizeSample: parseInt(
											e.currentTarget.value
										)
									})
								}
								class="select select-bordered w-full max-w-xs"
								value={resizeSample}
								required>
								<option>-1</option>
								<option selected>0</option>
								<option>1</option>
							</select>
						</div>
					)}
					<DirectoryInput
						path={dirPath}
						onChange={(path) =>
							this.setState({
								dirPath: path,
								file: ''
							})
						}
					/>
					<FileInput
						invalid={this.state.invalidInput}
						path={dirPath}
						onChange={(file) => {
							this.setState({
								file,
								portfolios: [],
								invalidInput: false,
								xHeader: undefined
							})
							try {
								if (dirPath == null) throw new Error()

								const wb = xlsx.readFile(
									path.join(dirPath, file)
								)

								this.rows = xlsx.utils.sheet_to_json<
									Record<string, any>
								>(wb.Sheets[wb.SheetNames[0]], { defval: null })

								const xh = Object.keys(this.rows[0]).map((x) =>
									x.trim()
								)

								if (
									!xh.includes('Date') &&
									!xh.includes('date')
								)
									throw new Error()

								this.setState({
									xHeader: xh,
									modelVars: []
								})
							} catch (e) {
								this.setState({
									invalidInput: true
								})
							}
						}}
						file={file}
					/>
					<div class="form-control">
						<label class="label">
							<span class="label-text">Portfolios</span>
							<FieldHelp id="portfhelp">
								<h1 class="font-bold text-xl">Portfolios</h1>
								<p>
									Please select one or more portfolios that
									are going to be used in the experiment.
								</p>
							</FieldHelp>
						</label>
						<input
							required
							disabled={xHeader == null}
							class="input input-bordered w-full"
							value={portfolios?.join(', ')}
						/>
						{xHeader && (
							<label
								for="portfolioModal"
								style={{
									position: 'absolute',
									right: 0,
									width: 200,
									height: 48,
									cursor: 'pointer'
								}}
							/>
						)}
					</div>
					<span class="performance">
						<Input
							title="Performance"
							type="range"
							help={
								<Fragment>
									<h1 class="font-bold text-xl">
										Performance
									</h1>
									<p>
										This slider determines how many threads
										the calculations will run on. With
										higher values the execution will be
										quicker, but general system performance
										will be degraded. Lower values are
										recommended if you want to use other
										programs while running.
									</p>
								</Fragment>
							}
							onInput={(e) =>
								(this.threads = parseInt(e.currentTarget.value))
							}
							defaultValue={this.threads.toString()}
							max={os.cpus().length}
							min={1}
							step={1}
							style={{
								padding: 0,
								height: 30,
								position: 'relative',
								top: 7
							}}
						/>
						<a>1</a>
						<a>{os.cpus().length}</a>
					</span>
					<Input
						title="No. Independent Variables"
						type="number"
						required
						min={1}
						defaultValue={
							(modelVars.length > 0 &&
								(modelVars.length + 1).toString()) ||
							undefined
						}
						onInput={(e) =>
							(this.independentVars = parseInt(
								e.currentTarget.value
							))
						}
						help={
							<Fragment>
								<h1 class="font-bold text-xl">
									No. Independent Variables
								</h1>
								<p>
									Please select the maximum number of
									Independent Variables that will be used when
									building the Forward Looking models. The
									default value is the number of selected
									Model Variables + 1.
								</p>
							</Fragment>
						}
					/>
					<div class="form-control">
						<label class="label">
							<span class="label-text">Hypothesis</span>
							<FieldHelp id="hypohelp">
								<h1 class="font-bold text-xl">Hypothesis</h1>
								<p>
									Please select the Hypothesis (the
									statistical assumptions) that you want to be
									used during the experiment. The Null
									Hypothesis is selected by default.
								</p>
							</FieldHelp>
						</label>
						<input
							required
							class="input input-bordered w-full"
							value={
								(hypo != null &&
									Object.values(hypo).join(' - ')) ||
								undefined
							}
						/>
						<label
							for="hypoModal"
							style={{
								position: 'absolute',
								right: 0,
								width: 200,
								height: 48,
								cursor: 'pointer'
							}}
						/>
					</div>
					<div class="form-control">
						<label class="label">
							<span class="label-text">Model Variables</span>
							<FieldHelp id="modelvarshelp">
								<h1 class="font-bold text-xl">
									Model Variables
								</h1>
								<p>
									Please select one or more exogenous
									variables that will be used to run the
									experiment.
								</p>
							</FieldHelp>
						</label>
						<input
							required
							disabled={xHeader == null}
							class="input input-bordered w-full"
							value={modelVars.join(', ')}
						/>
						{xHeader && (
							<label
								for="modelVarsModal"
								style={{
									position: 'absolute',
									right: 0,
									width: 200,
									height: 48,
									cursor: 'pointer'
								}}
							/>
						)}
					</div>
					<br />
					<label
						onClick={() => {
							this.runScript()
						}}
						class="btn btn-secondary"
						style={{
							marginTop: 20,
							width: 100
						}}>
						Run
					</label>
					<br />
					{dirPath != null && (
						<label
							for="docmodal"
							style={{
								lineHeight: 2.5,
								zIndex: 1,
								fontSize: 14
							}}
							class="link text-base-300">
							Generate Documentation
						</label>
					)}
				</form>
				<div
					style={{
						position: 'absolute',
						right: 30,
						top: 0,
						bottom: 0,
						width: 180,
						padding: 10,
						paddingTop: 15
					}}>
					<label class="label">
						<span class="label-text">Reporting Dates</span>
					</label>
					<div
						class="reportingDates card card-bordered p-4"
						style={{
							height: 380,
							overflow: 'auto'
						}}>
						{portfolios.map((p) => (
							<p>
								<span class="font-bold">{p}</span>
								<br />
								<span
									style={{
										opacity: 0.8
									}}>
									{this.calculatePortfolioDate(p)}
								</span>
							</p>
						))}
					</div>
				</div>
				<LogModal
					finished={finished}
					log={log}
					dirPath={dirPath}
					elapsed={elapsed}
					running={running || false}
					currentPortfolio={currentPortfolio}
					progress={
						(currentPortfolio != null &&
							portfolios.length > 1 &&
							`${
								portfolios.indexOf(currentPortfolio) + 1
							} out of ${portfolios.length}`) ||
						undefined
					}
					onKill={() => {
						this.setState({
							running: false
						})
						this.runner?.process?.kill()
					}}
				/>
				<DocModal baseDir={dirPath} />
				<HypoModal
					onHypo={(hypo) => {
						this.setState({
							hypo
						})
						;(
							document.querySelector(
								'#hypoModal'
							) as HTMLInputElement
						).checked = false
					}}
					mode={
						(this.state.mode != 'normal' && this.state.mode) ||
						undefined
					}
				/>
				<CheckboxModal
					id="modelVarsModal"
					choices={xHeader?.filter((h) => h.startsWith('x'))}
					current={modelVars}
					onAdd={(v) =>
						this.setState({
							modelVars: [...modelVars, v]
						})
					}
					onDel={(v) =>
						this.setState({
							modelVars: modelVars.filter((mv) => mv != v)
						})
					}
				/>
				<CheckboxModal
					id="portfolioModal"
					choices={xHeader?.filter(
						(h) =>
							!h.startsWith('x') && !(h.toLowerCase() == 'date')
					)}
					current={portfolios}
					onAdd={(v) => {
						const p = [...portfolios, v]
						this.setState({
							portfolios: p
						})
					}}
					onDel={(v) => {
						const p = portfolios.filter((mv) => mv != v)
						this.setState({
							portfolios: p
						})
					}}
				/>
			</Fragment>
		)
	}
}
