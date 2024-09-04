import { h } from 'preact'
import type RemoteType from '@electron/remote'
const remote: typeof RemoteType = require('@electron/remote')
import { IonIcon } from '@ionic/react'
import { folderOpenOutline } from 'ionicons/icons'
import FieldHelp from './FieldHelp'

interface Props {
	path?: string
	onChange: (path: string) => void
}

export default function DirectoryInput(props: Props) {
	return (
		<div
			class="form-control"
			style={{
				width: 'calc(100% - 20px)',
				display: 'inline-block',
				position: 'relative',
				margin: '0 10px'
			}}>
			<label class="label">
				<span class="label-text">Input Directory </span>
				<FieldHelp id="idhelp">
					<h1 class="font-bold text-xl">Input Directory</h1>
					<p>
						Please select the directory where the input file(s) can
						be found and where you want the output to be generated.
					</p>
				</FieldHelp>
			</label>
			<input
				readonly
				style={{
					paddingRight: 60
				}}
				type="text"
				value={props.path}
				class="input input-bordered w-full"
				onInput={(e) => props.onChange(e.currentTarget.value)}
			/>
			<div
				class="btn btn-sm btn-secondary"
				style={{
					position: 'absolute',
					right: 8,
					bottom: 6,
					fontSize: 16,
					height: 36,
					width: 36,
					padding: 0
				}}
				onClick={async () => {
					const path = await remote.dialog.showOpenDialog({
						properties: ['openDirectory']
					})
					if (path.canceled) return
					props.onChange(path.filePaths[0])
				}}>
				<IonIcon icon={folderOpenOutline} />
			</div>
		</div>
	)
}
