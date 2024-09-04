import { Fragment, h } from 'preact'
import { useState, useRef } from 'preact/hooks'
import PortfolioInput from './PortfolioInput'
import generateFullDocx from '../GenerateFullDocx'

export default function DocModal({ baseDir }: { baseDir?: string }) {
	const [portfolio, setPortfolio] = useState<string>()
	const [loading, setLoading] = useState(false)
	const modelRow = useRef(1)
	const comparisonModelRow1 = useRef(3)
	const comparisonModelRow2 = useRef(4)

	return (
		<Fragment>
			<input type="checkbox" id="docmodal" class="modal-toggle" />
			<label for="docmodal" class="modal">
				<label
					for=""
					class="modal-box docmodal"
					style={{
						textAlign: 'center'
					}}>
					<div
						style={{
							textAlign: 'left'
						}}>
						<PortfolioInput
							portfolio={portfolio}
							path={baseDir}
							onChange={(x) => setPortfolio(x)}
						/>
						<br />
						<div
							class="form-control"
							style={{
								display: 'inline-block',
								position: 'relative',
								width: 200,
								margin: '0 10px'
							}}>
							<label class="label">
								<span class="label-text">Champion Model Row #</span>
							</label>
							<input
								min={1}
								type="number"
								onChange={(e) =>
									(modelRow.current = Number(
										e.currentTarget.value
									))
								}
								required
								class={'input input-bordered w-full'}
							/>
							<label class="label">
								<span class="label-text">Comparison Model 1 Row #</span>
							</label>
							<input
								min={1}
								type="number"
								onChange={(e) =>
									(comparisonModelRow1.current = Number(
										e.currentTarget.value
									))
								}
								required
								class={'input input-bordered w-full'}
							/>
							<label class="label">
								<span class="label-text">Comparison Model 2 Row #</span>
							</label>
							<input
								min={1}
								type="number"
								onChange={(e) =>
									(comparisonModelRow2.current = Number(
										e.currentTarget.value
									))
								}
								required
								class={'input input-bordered w-full'}
							/>
						</div>
					</div>
					<label
						class="btn btn-secondary"
						onClick={async () => {
							setLoading(true)
							try {
								await generateFullDocx(baseDir!, [
									{
										dir: portfolio!,
										modelRow: modelRow.current,
										comparisonModelRow1: comparisonModelRow1.current,
										comparisonModelRow2: comparisonModelRow2.current
									}
								])
							} catch (error) {
								alert(error)
							} finally {
								setLoading(false)
							}
						}}
						style={{
							marginTop: 100
						}}>
						{(loading && '...') || 'generate'}
					</label>
				</label>
			</label>
		</Fragment>
	)
}
