import { IonIcon } from '@ionic/react'
import { add, checkmark, close, trashOutline } from 'ionicons/icons'
import { Component, Fragment, h } from 'preact'

export interface Hypothesis {
	name: string
	p_value_model: number
	p_value_blue: number
	lags: number
	stationarity: string
	mode?: string
}

export const defaultHypos = [
	{
		name: 'H0',
		p_value_model: 0.1,
		p_value_blue: 0.1,
		lags: 8,
		stationarity: 'N'
	},
	{
		name: 'H0',
		p_value_model: 0.1,
		p_value_blue: 0.1,
		lags: 8,
		stationarity: 'Y',
		mode: 'bt'
	}
]

interface Props {
	onHypo: (hypo: Hypothesis) => void
	mode?: string
}
interface State {
	hypos: Hypothesis[]
	adding?: boolean
}

const defaultAdding: Record<string, string> = {
	name: 'H0',
	p_value_model: '0.1',
	p_value_blue: '0.1',
	lags: '8',
	stationarity: 'N'
}

export default class HypoModal extends Component<Props, State> {
	adding: Record<string, string> = defaultAdding

	constructor(props: Props) {
		super(props)

		const h = localStorage.getItem('hypos')
		this.state = {
			hypos: (h != null && JSON.parse(h)) || defaultHypos
		}
	}

	render() {
		return (
			<Fragment>
				<input type="checkbox" id="hypoModal" class="modal-toggle" />
				<label for="hypoModal" class="modal">
					<label
						for=""
						style={{
							height: 400,
							maxWidth: 600,
							overflow: 'auto',
							paddingBottom: 80
						}}
						class="modal-box">
						<table class="table w-full">
							<thead>
								<tr>
									<th></th>
									<th>p_value_model</th>
									<th>p_value_blue</th>
									<th>Lags</th>
									<th>Station.</th>
									<th></th>
								</tr>
							</thead>
							<tbody>
								{this.state.hypos
									.filter((h) => h.mode == this.props.mode)
									.map((h) => (
										<tr
											style={{ cursor: 'pointer' }}
											onClick={() =>
												this.props.onHypo(h)
											}>
											<td>{h.name}</td>
											<td>{h.p_value_model}</td>
											<td>{h.p_value_blue}</td>
											<td>{h.lags}</td>
											<td>{h.stationarity}</td>
											<td>
												<button
													class="btn btn-ghost"
													onClick={() => {
														this.setState(
															{
																hypos: this.state.hypos.filter(
																	(x) =>
																		h != x
																)
															},
															() => {
																localStorage.setItem(
																	'hypos',
																	JSON.stringify(
																		this
																			.state
																			.hypos
																	)
																)
															}
														)
													}}>
													<IonIcon
														icon={trashOutline}
													/>
												</button>
											</td>
										</tr>
									))}
								{this.state.adding && (
									<tr>
										{Object.entries(this.adding)
											.slice(0, -1)
											.map(([k, v]) => (
												<td class="m-0 p-2">
													<input
														class="input input-bordered w-full p-2"
														defaultValue={v}
														onInput={(e) => {
															// @ts-ignore
															this.adding[k] =
																e.currentTarget.value
														}}
													/>
												</td>
											))}
										<td class="m-0 p-2">
											<input
												type="checkbox"
												defaultChecked={
													this.adding.stationarity ===
													'Y'
												}
												class="checkbox"
												onInput={(e) => {
													this.adding[
														'stationarity'
													] =
														(e.currentTarget
															.checked &&
															'Y') ||
														'N'
												}}
											/>
										</td>
									</tr>
								)}
								{(this.state.adding && (
									<tr>
										<button
											onClick={() => {
												const { adding } = this
												this.setState(
													{
														adding: false,
														hypos: [
															...this.state.hypos,
															{
																name: adding[
																	'name'
																],
																p_value_model:
																	parseFloat(
																		adding[
																			'p_value_model'
																		]
																	),
																p_value_blue:
																	parseFloat(
																		adding[
																			'p_value_blue'
																		]
																	),
																lags: parseInt(
																	adding[
																		'lags'
																	]
																),
																stationarity:
																	adding[
																		'stationarity'
																	],
																mode: this.props
																	.mode
															}
														]
													},
													() => {
														localStorage.setItem(
															'hypos',
															JSON.stringify(
																this.state.hypos
															)
														)
													}
												)
											}}
											style={{
												position: 'absolute',
												paddingLeft: 30,
												textTransform: 'none'
											}}
											class="btn btn-secondary btn-ghost text-primary-content">
											<IonIcon
												icon={checkmark}
												style={{
													display: 'inline-block',
													position: 'absolute',
													left: 10
												}}
											/>
											Save
										</button>
										<button
											onClick={() => {
												this.setState({
													adding: false
												})
											}}
											style={{
												position: 'absolute',
												paddingLeft: 30,
												left: 90,
												textTransform: 'none'
											}}
											class="btn btn-secondary btn-ghost text-error">
											<IonIcon
												icon={close}
												style={{
													display: 'inline-block',
													position: 'absolute',
													left: 10
												}}
											/>
											Cancel
										</button>
									</tr>
								)) || (
									<tr>
										<button
											onClick={() => {
												this.adding = defaultAdding

												this.setState({
													adding: true
												})
											}}
											style={{
												position: 'absolute',
												paddingLeft: 30,
												textTransform: 'none'
											}}
											class="btn btn-secondary btn-ghost text-primary-content">
											<IonIcon
												icon={add}
												style={{
													display: 'inline-block',
													position: 'absolute',
													left: 10
												}}
											/>
											Add Hypothesis
										</button>
										<label
											style={{
												position: 'absolute',
												right: 0
											}}
											for="hypoModal"
											class="btn btn-error">
											Close
										</label>
									</tr>
								)}
							</tbody>
						</table>
					</label>
				</label>
			</Fragment>
		)
	}
}
