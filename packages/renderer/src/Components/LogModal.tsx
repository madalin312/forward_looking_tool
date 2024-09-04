import { Fragment, h } from 'preact'
import reactStringReplace from 'react-string-replace'
import type RemoteType from '@electron/remote'
const remote: typeof RemoteType = require('@electron/remote')

export function secondsToTime(secs: number) {
	const minutes = Math.floor(secs / 60)
	const seconds = secs % 60

	return `${minutes}m${seconds}s`
}

export default function LogModal({
	log,
	dirPath,
	elapsed,
	running,
	onKill,
	currentPortfolio,
	finished,
	progress
}: {
	log: string[]
	dirPath?: string
	elapsed: number
	running: boolean
	finished?: boolean
	onKill: () => void
	currentPortfolio?: string
	progress?: string
}) {
	return (
		<Fragment>
			<input
				type="checkbox"
				id="logModal"
				class="modal-toggle"
				checked={running}
			/>
			<div class="modal">
				<label
					for=""
					class="modal-box bg-neutral logModal"
					style={{
						padding: 0,
						color: '#ccc'
					}}>
					<div
						style={{
							overflow: 'auto',
							height: 300,
							display: 'flex',
							flexDirection: 'column-reverse',
							padding: '0 20px'
						}}>
						<div>
							{log.slice(-2000).map((l) => (
								<pre
									style={{
										overflowWrap: 'anywhere',
										whiteSpace: 'pre-line',
										...((l.charAt(0) == '!' && {
											color: 'hsl(var(--er))'
										}) ||
											undefined)
									}}>
									{reactStringReplace(
										l.replace(/^!/, ''),
										/{link}/,
										() => (
											<a
												onClick={() => {
													if (dirPath)
														remote.shell.openPath(
															dirPath
														)
												}}
												href="javascript:void"
												style={{
													textDecoration: 'underline'
												}}>
												Open Folder
											</a>
										)
									)}
								</pre>
							))}
						</div>
					</div>
					<div
						style={{
							height: 52,
							position: 'relative'
						}}
						class="bg-neutral-focus">
						<code
							style={{
								position: 'absolute',
								left: 20,
								top: 15
							}}>
							{(currentPortfolio && currentPortfolio + ' | ') ||
								''}
							{(progress && progress + ' | ') || ''}
							{secondsToTime(elapsed)} elapsed
						</code>
						{(finished && (
							<button
								class="btn btn-sm"
								onClick={() => onKill()}
								style={{
									position: 'absolute',
									right: 10,
									top: 10
								}}>
								close
							</button>
						)) || (
							<label
								for="confirmExitModal"
								class="btn btn-error btn-sm"
								style={{
									position: 'absolute',
									right: 10,
									top: 10
								}}>
								end script
							</label>
						)}
					</div>
				</label>
			</div>
			<input type="checkbox" id="confirmExitModal" class="modal-toggle" />
			<div class="modal">
				<div
					class="modal-box"
					style={{
						width: 300
					}}>
					<p>
						Are you sure you want to end the currently running
						script?
					</p>
					<div class="modal-action">
						<label
							for="confirmExitModal"
							onClick={() => onKill()}
							class="btn btn-error">
							Yes
						</label>
						<label for="confirmExitModal" class="btn">
							No
						</label>
					</div>
				</div>
			</div>
		</Fragment>
	)
}
