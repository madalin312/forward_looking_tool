import { Fragment, h } from 'preact'
import { useState, useRef } from 'preact/hooks'
import PortfolioInput from './PortfolioInput'
import runSensitivityTest from '../RunSensitivityTest'
import FieldHelp from '../Components/FieldHelp'

export default function SensitivityModal({ baseDir }: { baseDir?: string }) {
	const [portfolio, setPortfolio] = useState<string>()
	const [loading, setLoading] = useState(false)
	const [mode, setMode] = useState<string>('normal') // Add mode state
	const [resizeSample, setResizeSample] = useState<number>(0) // Add resizeSample state
	const modelRow = useRef(1)
	const increasePercentage = useRef(3)

	return (
		<Fragment>
			<input type="checkbox" id="sensitivitymodal" class="modal-toggle" />
			<label for="sensitivitymodal" class="modal">
				<label
					for=""
					class="modal-box docmodal"
					style={{
						textAlign: 'center'
					}}>
					<h2>Sensitivity test</h2>
					<div
						style={{
							textAlign: 'left',
							width: 400,
							marginLeft: 25,
							margin: '0 auto'
						}}>

						<div
							style={{
								display: 'flex',
								justifyContent: 'center',
								gap: '20px',
								marginTop: '20px',
								marginLeft: '10px'
							}}>
							<div>
								<label class="label">
									<span class="label-text">Entity Type</span>
									<FieldHelp id="sensitivitymodehelp">
										<h1 class="font-bold text-xl">Entity Type</h1>
										<p>
											Choose the type of the entity. Currently
											implemented types are Normal and BT (Banca
											Transilvania)
										</p>
									</FieldHelp>
								</label>
								<select
									onInput={(e) => setMode(e.currentTarget.value)}
									class="select select-bordered w-full max-w-xs"
									value={mode}
									required>
									<option value="normal">Normal</option>
									<option value="bt">BT</option>
								</select>
							</div>

							{mode == 'bt' && (
								<div>
									<label class="label">
										<span class="label-text">Resize Sample</span>
										<FieldHelp id="resizesamplehelp">
											<h1 class="font-bold text-xl">Resize Sample</h1>
											<p>
												This parameter is used to establish if
												linear interpolated data is used.
												Default is 0, meaning the first row is
												not considered in development
											</p>
										</FieldHelp>
									</label>
									<select
										onInput={(e) => setResizeSample(parseInt(e.currentTarget.value))}
										class="select select-bordered w-full max-w-xs"
										value={resizeSample}
										required>
										<option value={-1}>-1</option>
										<option value={0}>0</option>
										<option value={1}>1</option>
									</select>
								</div>
							)}
						</div>
						<PortfolioInput
							portfolio={portfolio}
							path={baseDir}
							onChange={(x) => setPortfolio(x)}
						/>
						<div
							style={{
								display: 'flex',
								justifyContent: 'left-right',
								gap: '20px',
								marginTop: '20px',
								marginLeft: '0px'
							}}>
							<div
								style={{
									flex: '1',
									width: '50%',
								}}>
								<label class="label"
									style={{
										margin: '0 10px'
									}}>
									<span class="label-text">Model row #</span>
								</label>
								<input
									min={2}
									type="number"
									onChange={(e) =>
									(modelRow.current = Number(
										e.currentTarget.value
									))
									}
									required
									class={'input input-bordered w-full'}
									style={{
										margin: '0 10px'
									}}
								/>
							</div>
							<div>
								<label class="label"
									style={{
										margin: '0 10px',
										width: '100%'
									}}>
									<span class="label-text">Increase/Decrease percentage</span>
								</label>
								<input
									min={1}
									type="number"
									onChange={(e) =>
									(increasePercentage.current = Number(
										e.currentTarget.value
									))
									}
									required
									class={'input input-bordered w-full'}
									style={{
										margin: '0 10px'
									}}
								/>
							</div>
						</div>
					</div>
					<label
						class="btn btn-secondary"
						onClick={async () => {
							setLoading(true)
							try {
								await runSensitivityTest(
									baseDir!,
									{
										dir: portfolio!,
										modelRow: modelRow.current,
										increasePercentage: increasePercentage.current
									},
									mode,
									resizeSample
								)
							} catch (error) {
								alert(error)
							} finally {
								setLoading(false)
							}
						}}
						style={{
							marginTop: 20
						}}>
						{(loading && '...') || 'generate'}
					</label>
				</label>
			</label>
		</Fragment>
	)
}
