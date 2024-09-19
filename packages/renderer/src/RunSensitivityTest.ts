import fs from 'fs/promises'
import path from 'path'
import type XlsxType from 'xlsx'
import { spawn } from 'child_process'
import { venvPath } from './RunPythonScript'

export default async function runSensitivityTest(
	baseDir: string,
	portfolio: {
		dir: string
		modelRow: number
		increasePercentage: number
	},
	mode: string,
	resizeSample: number
) {
	let isBuilt = true

	try {
		await fs.stat(path.join(process.cwd(), 'resources/app'))
	} catch (e) {
		isBuilt = false
	}

	const scriptPath = path.join(
		(isBuilt && path.join(process.cwd(), 'resources', 'app')) ||
		process.cwd(),
		'python'
	)
	
	portfolio.modelRow -= 2 // Subtract 2 to adjust for zero-based index and for excel file header

	let scriptFile = 'sensitivity_test.py'
		switch (mode) {
			case 'bt':
				scriptFile = 'sensitivity_test_BT.py'
				break
			default:
				break
		}

	await new Promise<void>((resolve, reject) => {
		const process = spawn(
			path.join(venvPath, 'Scripts', 'python.exe'),
			[
				path.join(scriptPath, scriptFile),
				baseDir,
				portfolio.dir,
				portfolio.modelRow.toString(),
				portfolio.increasePercentage.toString(),
				resizeSample.toString()
			]
		)
		process.on('close', (exitCode) => {
			if (exitCode === 0) {
				alert("Process closed successfully");
				resolve();
			} else {
				alert(`Process exited with code ${exitCode}`);
				reject(new Error(`Process exited with code ${exitCode}`));
			}
		});
	
		process.stderr.on('data', (data) => {
			alert(`stderr: ${data.toString()}`);
			reject(new Error(data.toString()));
		});
	
		process.on('error', (err) => {
			alert(`Failed to start process: ${err.message}`);
			reject(err);
		});
	})
}
