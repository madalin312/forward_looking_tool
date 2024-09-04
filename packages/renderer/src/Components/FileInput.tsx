import { Component, createRef, h } from 'preact'
import fs from 'fs/promises'
import FieldHelp from './FieldHelp'

interface Props {
	path?: string
	onChange: (path: string) => void
	file?: string
	invalid?: boolean
}
interface State {
	files?: string[]
}

export default class FileInput extends Component<Props, State> {
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
					<span class="label-text">Input Filename</span>
					<FieldHelp id="ifhelp">
						<h1 class="font-bold text-xl">Input Filename</h1>
						<p>
							Please select the input file that you want to be
							used in the experiment.
						</p>
					</FieldHelp>
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
								files: (
									await fs.readdir(this.props.path, {
										withFileTypes: true
									})
								)
									.filter(
										(f) =>
											f.isFile() &&
											!f.name.startsWith('~') &&
											f.name.endsWith('.xlsx')
									)
									.map((f) => f.name)
							})
						}}
						value={this.props.file}
					/>
					<ul
						ref={this.ref}
						style={{
							marginTop: 10
						}}
						tabIndex={0}
						class="dropdown-content menu p-2 shadow bg-base-100 rounded-box dropdownOptions">
						{this.state.files?.map((f) => (
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
