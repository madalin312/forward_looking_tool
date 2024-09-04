import { Component, createRef, h } from 'preact'
import fs from 'fs/promises'

interface Props {
	path?: string
	onChange: (path: string) => void
	portfolio?: string
	invalid?: boolean
}
interface State {
	portfolios?: string[]
}

export default class PortfolioInput extends Component<Props, State> {
	ref = createRef<HTMLUListElement>()
	render() {
		return (
			<div
				class="form-control"
				style={{
					display: 'inline-block',
					position: 'relative',
					width: 200,
					margin: '0 10px'
				}}>
				<label class="label">
					<span class="label-text">Portfolio</span>
				</label>
				<div class="dropdown dropdown-bottom dropdown-end">
					<input
						required
						tabIndex={0}
						class={
							'input input-bordered w-full' +
							((this.props.invalid && ' input-error') || '')
						}
						disabled={this.props.path == null}
						onClick={async () => {
							if (this.props.path == null) return
							this.setState({
								portfolios: (
									await fs.readdir(this.props.path, {
										withFileTypes: true
									})
								)
									.filter((f) => f.isDirectory())
									.map((f) => f.name)
							})
						}}
						value={this.props.portfolio}
					/>
					<ul
						ref={this.ref}
						style={{
							marginTop: 10
						}}
						tabIndex={0}
						class="dropdown-content menu p-2 shadow bg-base-100 rounded-box dropdownOptions">
						{this.state.portfolios?.map((f) => (
							<li>
								<a
									onClick={() => {
										this.props.onChange(f)

										this.ref.current?.blur()
									}}>
									{f}
								</a>
							</li>
						))}
					</ul>
				</div>
			</div>
		)
	}
}
