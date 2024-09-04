import { h, VNode } from 'preact'
import FieldHelp from './FieldHelp'

export default function Input(
	props: {
		title: string
		help: VNode<any>
	} & h.JSX.HTMLAttributes<HTMLInputElement>
) {
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
				<span class="label-text">{props.title}</span>
				<FieldHelp
					id={props.title.toLowerCase().replaceAll(' ', '') + 'help'}>
					{props.help}
				</FieldHelp>
			</label>
			<input {...props} class="input input-bordered w-full" />
		</div>
	)
}
