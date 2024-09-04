import { IonIcon } from '@ionic/react'
import { helpCircle, helpCircleOutline } from 'ionicons/icons'
import { ComponentChildren, Fragment, h } from 'preact'

export default function FieldHelp({
	children,
	id
}: {
	id: string
	children: ComponentChildren
}) {
	return (
		<Fragment>
			<label
				for={id}
				class="text-info-content cursor-pointer"
				style={{
					fontSize: 18,
					opacity: 0.3,
					position: 'absolute',
					right: 5,
					marginBottom: -7
				}}>
				<IonIcon icon={helpCircleOutline} />
			</label>
			<input type="checkbox" id={id} class="modal-toggle" />
			<label for={id} class="modal">
				<label
					class="modal-box relative text-left"
					for=""
					style={{
						width: 400
					}}>
					{children}
					<div class="modal-action">
						<label for={id} class="btn">
							Dismiss
						</label>
					</div>
				</label>
			</label>
		</Fragment>
	)
}
