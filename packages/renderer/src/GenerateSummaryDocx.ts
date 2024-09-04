import {
	Document,
	Packer,
	Paragraph,
	Table,
	TableCell,
	TableRow,
	TextRun,
	AlignmentType,
	HeadingLevel,
	Alignment
} from 'docx'
import { table } from 'node:console'

export default async function generateSummaryDocx(summary: string) {
	let lines = summary.split(/[\n\r]/)

	const firstLine = lines.filter((l) => l.length != 0).shift()
	if (firstLine == null) throw new Error()

	lines = lines.slice(lines.indexOf(firstLine) + 1)

	const children: any[] = []

	let inTable = false
	let previousLine = ''
	let tableRows: string[][] = []

	const tableCaptions: string[] = ["Table 3 Model Definition Summary", "Table 4 BLUE Tests Summary", "Table 5 Significance Summary"];

	for (const i in lines) {
		let l = lines[i]
		if (l.trim() == 'Info') inTable = true

		if (tableCaptions.includes(l)) {

			const tableCaption = new Paragraph({
				children: [
					new TextRun({
						text: l,
						italics: true,
						size: 20,
						font: 'Calibri'
					}),
				],
				alignment: AlignmentType.JUSTIFIED,
			});

			children.push(tableCaption)

			continue
		}

		if (inTable) {
			const cols = l.match(/(.+)\s{5,}(.+)/)
			if (cols != null && cols[2] != 'Info') tableRows.push([cols[1], cols[2]])
			// else if ((cols != null && cols[2] == 'Info') || cols == null) continue

			if (
				Number(i) + 1 === lines.length ||
				(l.length == 0 && previousLine.length == 0)
			) {

				const rows = tableRows.map(
					(row) =>
						new TableRow({
							children: row.map(
								(col) =>
									new TableCell({
										children: [
											new Paragraph({
												children: [
													new TextRun({
														text: col,
														size: 24,
														font: 'Calibri'
													})
												],
												alignment: AlignmentType.JUSTIFIED
											})
										]
									})
							)
						})
				)
				children.push(
					new Table({
						rows
					})
				)

				inTable = false
				tableRows = []
			}
			previousLine = l

			// const cols = l.match(/(.+)\s{5,}(.+)/)

			// if (cols == null || cols[2] == 'Info') continue

			// tableRows.push([cols[1], cols[2]])

			continue
		}

		if (l.length === 0 && previousLine.length === 0) continue
		previousLine = l

		const isBullet = l.startsWith('*')
		if (isBullet) l = l.slice(2)
		children.push(
			new Paragraph({
				children: [
					new TextRun({
						text: l,
						size: 24,
						font: 'Calibri'
					})
				],
				bullet:
					(isBullet && {
						level: 0
					}) ||
					undefined,
				alignment: AlignmentType.JUSTIFIED
			})
		)
	}

	const doc = new Document({
		sections: [
			{
				children
			}
		]
	})

	return await Packer.toBuffer(doc)
}
