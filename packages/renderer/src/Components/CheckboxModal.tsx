import { Fragment, h } from 'preact'

interface Props {
	id: string
	choices?: string[]
	current: string[]
	onAdd: (mv: string) => void
	onDel: (mv: string) => void
}

export default function CheckboxModal(props: Props) {
	return (
		<Fragment>
			<input type="checkbox" id={props.id} class="modal-toggle" />
			<label for={props.id} class="modal">
				<label for="" class="modal-box checkboxModal">
					{props.choices?.map((c) => (
						<div class="form-control">
							<label class="label cursor-pointer">
								<input
									type="checkbox"
									class="checkbox"
									onChange={(e) => {
										if (e.currentTarget.checked)
											props.onAdd(c)
										else props.onDel(c)
									}}
									checked={props.current.includes(c)}
								/>
								<span class="label-text">{c}</span>
							</label>
						</div>
					))}
					<br />
					<label
						for={props.id}
						class="btn btn-secondary"
						style={{
							marginTop: 20
						}}>
						OK
					</label>
				</label>
			</label>
		</Fragment>
	)
}
