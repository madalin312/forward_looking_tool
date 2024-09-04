import fs from 'fs/promises'
import path from 'path'
import PizZip from 'pizzip'
import Docxtemplater from 'docxtemplater'
import type XlsxType from 'xlsx'
import { pick } from 'lodash'
import os from 'os'
import { spawn } from 'child_process'
import generateSummaryDocx from './GenerateSummaryDocx'
import { venvPath } from './RunPythonScript'
const xlsx: typeof XlsxType = require('xlsx')

export default async function generateFullDocx(
	baseDir: string,
	portfolios: {
		dir: string
		modelRow: number
		comparisonModelRow1: number
		comparisonModelRow2: number
	}[]
) {
	let isBuilt = true

	try {
		await fs.stat(path.join(process.cwd(), 'resources/app'))
	} catch (e) {
		isBuilt = false
	}

	const templatesPath = path.join(
		(isBuilt && path.join(process.cwd(), 'resources', 'app')) ||
			process.cwd(),
		'python'
	)

	let segmentidx = 0
	const incrSegment = () => {
		segmentidx++
		return segmentidx
	}

	const tempPath = await fs.mkdtemp(path.join(os.tmpdir(), 'fwldg-'))

	const extractSubstringFromFile = async (filePath: string): Promise<string> => {
		try {
			const content = await fs.readFile(filePath, 'utf8');
			const regex = /\*\*\[(.*?)\]\*\*/;
			const match = content.match(regex);
			return match ? match[1] : 'N/A';
		} catch (error) {
			console.error('Could not find executive summary\nError reading file:', error);
			return 'N/A';
		}
	};

	function transposeJsonArray(jsonArray: Record<string, any>[]): Record<string, any>[] {
		if (jsonArray.length === 0) {
			return [];
		}
		
		let transposed: Record<string, any>[] = [];
		let allKeys = new Set<string>();

		// Collect all keys (column headers)
		jsonArray.forEach(item => {
			Object.keys(item).forEach(key => allKeys.add(key));
		});
	
		if (allKeys.size == 0) {
			throw new Error("No columns in the json array!")
		}

		// Initialize object for each key
		allKeys.forEach(() => {
			transposed.push({})
		});
		
		let keyIndex:number = 0
		allKeys.forEach((key) => {
			jsonArray.forEach((row, arrayIndex) => {
				transposed[keyIndex][arrayIndex.toString()] = row[key] ?? 'N/A';
			});
			keyIndex++;
		});
	
		return transposed;
	}

	function describeVariable(variable: string, variableDescription:Record<string, any>) {
		let inputVariable = variable
		variable = variable.replace("diff_diff", "diff2")

		let components: string[] = variable.split("_")
		let descriptions: string[] = []
		components.forEach((component, index) => {
			descriptions.push(variableDescription[component] === undefined && index == 0 ? "portfolio" : variableDescription[component])
		})
		descriptions.reverse()

		let fullDescription = inputVariable + " represents "
		
		descriptions.forEach((description, index) => {
			if(index == 0)
				fullDescription = fullDescription + description
			else if(index == descriptions.length-1)
				fullDescription = fullDescription + " of the " + description
			else
				fullDescription = fullDescription + " of " + description
		})
		return fullDescription
	}

	function selectColumnsFromJsonArray(jsonArray: Record<string, any>[], columns:string[]): Record<string, any>[] {
		const subArray = jsonArray.map(item => {
			let newItem: Record<string, any> = {};
		
			columns.forEach(key => {
				if (key in item) {
					newItem[key] = item[key];
				}
			});
		
			return newItem;
		});
		
		return subArray
	}

	for (const p of portfolios) {
		p.modelRow -= 2
		p.comparisonModelRow1 -= 2
		p.comparisonModelRow2 -= 2
		const portfolio = p.dir.split('_H').slice(0, -1).join('_H')
		const dir = path.join(baseDir, p.dir)
		const files = await fs.readdir(dir)

		let summary = ''

		await new Promise<void>((resolve, reject) => {
			const process = spawn(
				path.join(venvPath, 'Scripts', 'python.exe'),
				[
					path.join(templatesPath, 'switchsummarymodel.py'),
					dir,
					p.modelRow.toString()
				]
			)
			process.on('close', (e) => {
				if (process?.exitCode === 0) resolve()
				else reject(e)
			})
			process.stdout.on('data', (x) => {
				summary += x.toString()
			})
			process.on('error', (e) => {
				reject(e)
			})
		})

		summary = summary.trim()

		const executiveSummaryFile = files.find(file => file.startsWith("executive_summary_model_"))
		let originalIndVar:string = 'N/A'
		if (executiveSummaryFile == undefined) {
			console.error('Could not find executive summary in the portfolio folder');
        }
		else{
			await extractSubstringFromFile(path.join(dir, executiveSummaryFile))
				.then(extractedSubstring => {
					originalIndVar = extractedSubstring
				})

				.catch(error => {
					console.error('An error occurred:', error);
				});
		}

		const getExcel = (prefix: string) =>
			path.join(dir, files.find((f) => f.startsWith(prefix))!)

		const readExcel = (filePath: string) => {
			const wb = xlsx.readFile(filePath)
			return xlsx.utils.sheet_to_json<Record<string, any>>(
				wb.Sheets[wb.SheetNames[0]],
				{ defval: null }
			)
		}

		const mkS = (val?: number) => {
			if (val == null) return ''
			if (val < 0.01) return '***'
			if (val < 0.05) return '**'
			if (val < 0.1) return '*'
			return ''
		}

		const stationarity = readExcel(getExcel('04.'))
		const finalSubset = readExcel(getExcel('08.'))
		const final = readExcel(getExcel('12.'))
		const variableDescription = readExcel(path.join(templatesPath, 'transformation_variable_description_v2.xlsx'))[0]
		variableDescription["diff2"] = "the second absolute difference"
		const championModel = final[p.modelRow]
		const comparisonModel1 = final[p.comparisonModelRow1]
		const comparisonModel2 = final[p.comparisonModelRow2]
		const finalIndependent = Array.from(
			(championModel['Independent'] as string).matchAll(/'(.+?)'[,\]]/g)
		).map((championModel) => championModel[1])
		const comparisonModel1Independent = Array.from(
			(comparisonModel1['Independent'] as string).matchAll(/'(.+?)'[,\]]/g)
		).map((comparisonModel1) => comparisonModel1[1])
		const comparisonModel2Independent = Array.from(
			(comparisonModel2['Independent'] as string).matchAll(/'(.+?)'[,\]]/g)
		).map((comparisonModel2) => comparisonModel2[1])
		
		championModel['Number_of_ind_variables'] = finalIndependent.length
		comparisonModel1['Number_of_ind_variables'] = comparisonModel1Independent.length
		comparisonModel2['Number_of_ind_variables'] = comparisonModel2Independent.length

		championModel['Coeficient_1'] = finalIndependent[0] != '' ? championModel[finalIndependent[0]] : 'N/A'
		comparisonModel1['Coeficient_1'] = comparisonModel1Independent[0] != '' ? comparisonModel1[comparisonModel1Independent[0]] : 'N/A'
		comparisonModel2['Coeficient_1'] = comparisonModel2Independent[0] != '' ? comparisonModel2[comparisonModel2Independent[0]] : 'N/A'

		championModel['Coeficient_2'] = finalIndependent[1] != '' ? championModel[finalIndependent[1]] : 'N/A'
		comparisonModel1['Coeficient_2'] = comparisonModel1Independent[1] != '' ? comparisonModel1[comparisonModel1Independent[1]] : 'N/A'
		comparisonModel2['Coeficient_2'] = comparisonModel2Independent[1] != '' ? comparisonModel2[comparisonModel2Independent[1]] : 'N/A'

		championModel['Coeficient_3'] = finalIndependent[2] != '' ? championModel[finalIndependent[2]] : 'N/A'
		comparisonModel1['Coeficient_3'] = comparisonModel1Independent[2] != '' ? comparisonModel1[comparisonModel1Independent[2]] : 'N/A'
		comparisonModel2['Coeficient_3'] = comparisonModel2Independent[2] != '' ? comparisonModel2[comparisonModel2Independent[2]] : 'N/A'

		let finalModelComparisonColumns = [
			'Model_number',
			'Dependent',
			'Independent',
			'Intercept',
			'Coeficient_1',
			'Coeficient_2',
			'Coeficient_3',
			'*.model_pvalue',
			'R_Intercept_new_pvalue',
			'R_var1_new_pvalue',
			'R_var2_new_pvalue',
			'R_var3_new_pvalue',
			'*.adjRsq',
			'R_AICc',
			'Number_of_ind_variables'
		]

		let finalModelColumnHeaderRow = {
			'Model_number': 'Model_number',
			'Dependent': 'Dependent',
			'Independent': 'Independent',
			'Intercept': 'Intercept',
			'Coeficient_1': 'Coeficient_1',
			'Coeficient_2': 'Coeficient_2',
			'Coeficient_3': 'Coeficient_3',
			'*.model_pvalue': '*.model_pvalue',
			'R_Intercept_new_pvalue': 'R_Intercept_new_pvalue',
			'R_var1_new_pvalue': 'R_var1_new_pvalue',
			'R_var2_new_pvalue': 'R_var2_new_pvalue',
			'R_var3_new_pvalue': 'R_var3_new_pvalue',
			'*.adjRsq': '*.adjRsq',
			'R_AICc': 'R_AICc',
			'Number_of_ind_variables': 'Number_of_ind_variables'
		}

		let modelsToCompare = selectColumnsFromJsonArray([finalModelColumnHeaderRow, championModel, comparisonModel1, comparisonModel2], finalModelComparisonColumns)
		let transposedModelComparisonJsonArray = transposeJsonArray(modelsToCompare)

		const bestAicc = final.sort((a, b) => a['R_AICc'] - b['R_AICc'])[0]
		const bestRsq = final.sort((a, b) => b['*.adjRsq'] - a['*.adjRsq'])[0]
		const vifFail = final.filter((r) => r['VIF_test'] != 'Pass')
		const predFailCount = final.filter(
			(r) => r['Y_pred_check'] != 'OK'
		).length

		let H1H2Paragraph: string = ''
		if(dir.endsWith('H1')) {
			H1H2Paragraph = 'After applying all the filters from step 5, we identified no feasible models. We proceeded by considering an alternative hypothesis (H1) as per Step 6 presented above in the methodology section. The development followed the Steps from 2 to 5 under the new hypothesis.'
		}
		else if(dir.endsWith('H2')){
			H1H2Paragraph = 'After applying all the filters from step 5, we identified no feasible models. We proceeded by considering an alternative hypothesis (H1) as per Step 6 presented above in the methodology section. The development followed the Steps from 2 to 5 under the new hypothesis.\n\n'
			H1H2Paragraph += 'As we identified no feasible model under H1 hypothesis, we proceeded with the second alternative option: H2 as per Step 6 presented above in the methodology section. The development followed the Steps from 2 to 5 under the new hypothesis.'
		}

		let variableDescriptions: string = ''
		finalIndependent.forEach(independentVar => {
			variableDescriptions = variableDescriptions + describeVariable(independentVar, variableDescription) + ". "
		})

		const fmtTemplate = async (
			template: string,
			outputPath: string
			// ,writeExcel?: boolean
		) => {
			const content = await fs.readFile(
				path.join(templatesPath, template),
				'binary'
			)
			let zip = new PizZip(content)
			const doc = new Docxtemplater(zip, {
				paragraphLoop: true,
				linebreaks: true
			})

			doc.render(
				Object.fromEntries(
					Object.entries({
						adfKpssPpTrue: stationarity.filter(
							(s) => s['ADF_KPSS_PP'] === true
						).length,
						finalSubsetModelCount: finalSubset.length,
						bestModelIndependentVars: finalSubset[0]['Independent'],
						originalIndVars: originalIndVar,
						H1H2Paragraphs : H1H2Paragraph,
						coefTable: [
							{
								name: 'Intercept',
								estimate: championModel['Intercept'],
								pv: championModel['Intercept_pvalue'],
								s: mkS(championModel['Intercept_pvalue'])
							},
							...Array.from({
								length: 3
							})
								.map((_, i) => {
									const pv = championModel['*.var' + (i + 1) + '_pvalue']
									return {
										name: finalIndependent[i],
										estimate: championModel[finalIndependent[i]],
										pv,
										s: mkS(pv)
									}
								})
								.filter((x) => x.name != null)
						],
						modelPvalue: championModel['*.model_pvalue'],
						var1VIF: championModel['*.var1_VIF'],
						var2VIF: championModel['*.var2_VIF'],
						var3VIF: championModel['*.var3_VIF'],
						comb1: championModel['*.comb1_Corr_Pearson'],
						comb2: championModel['*.comb2_Corr_Pearson'],
						comb3: championModel['*.comb3_Corr_Pearson'],
						bg1: championModel['BG1_Lagrange_Multiplier_pvalue'],
						bg2: championModel['BG2_Lagrange_Multiplier_pvalue'],
						bg3: championModel['BG3_Lagrange_Multiplier_pvalue'],
						bg4: championModel['BG4_Lagrange_Multiplier_pvalue'],
						bg1test: championModel['BG1_test'],
						bg2test: championModel['BG2_test'],
						bg3test: championModel['BG3_test'],
						bg4test: championModel['BG4_test'],
						firstIndependentName: finalIndependent[0],
						secondIndependentName: finalIndependent[1],
						thirdIndependentName: finalIndependent[2],
						dw: championModel['R_DW_stat'],
						dwtest: championModel['R_DW_test'],
						sw: championModel['SW_pvalue'],
						swtest: championModel['SW_test'],
						jb: championModel['JB_pvalue'],
						jbtest: championModel['JB_test'],
						bp: championModel['BP_Lagrange_Multiplier_pvalue'],
						bptest: championModel['BP_test'],
						wh: championModel['R_WH_pvalue'],
						whtest: championModel['R_WH_test'],
						isInconsistent: [
							championModel['BP_test'],
							championModel['R_WH_TEST'],
							championModel['R_DW_TEST'],
							championModel['BG1_test'],
							championModel['BG2_test'],
							championModel['BG3_test'],
							championModel['BG4_test']
						].includes('Fail'),
						nwTable: [
							{
								name: 'Intercept',
								pv: championModel['R_Intercept_new_pvalue'],
								s: mkS(championModel['R_Intercept_new_pvalue'])
							},
							...Array.from({
								length: 3
							})
								.map((_, i) => {
									const pv =
										championModel['R_var' + (i + 1) + '_new_pvalue']
									return {
										name: finalIndependent[i],
										pv,
										s: mkS(pv)
									}
								})
								.filter((x) => x.name != null)
						],
						bestAiccNumber: bestAicc['Model_number'],
						bestRsqNumber: bestRsq['Model_number'],
						bestRsq: bestRsq['*.adjRsq'],
						bestAicc: bestAicc['R_AICc'],
						vifPass: vifFail.length === 0,
						vifFailCount: vifFail.length,
						insignificantModelCount: final.filter((r) =>
							[
								r['R_NW_test_var1'],
								r['R_NW_test_var2'],
								r['R_NW_test_var3']
							].includes('Not significant')
						).length,
						predPass: predFailCount === 0,
						predFailCount,
						threeVarModel: finalIndependent.length === 3,
						finalModelNumber: championModel['Model_number'],
						finalIndependentVars: finalIndependent.join(', '),
						finalIndependentVarCount: finalIndependent.length,
						modelComparisonTable: transposedModelComparisonJsonArray,
						independentVariableDescriptions: variableDescriptions,// JSON.stringify(variableDescription),
						stationarityFileName : path.basename(getExcel('04')),
						finalSubsetFileName : path.basename(getExcel('08')),
						finalFeasibleFileName : path.basename(getExcel('12')),
						championModelRow: p.modelRow,
						comparisonModelRow1: p.comparisonModelRow1,
						comparisonModelRow2: p.comparisonModelRow2,
						dependent: championModel['Dependent'],
						portfolio
					}).map(([k, v]) => [k, (v == null && 'N/A') || v])
				)
			)

			zip = doc.getZip()
			
			// This section embeds the 4, 8 and 12 files into the word document. Right now, we will choose a different approach
			// where we only leave the file names inside an annex in the word file, but we will keep the code here for future potential uses.
			// if (writeExcel) {
			// 	zip.file(
			// 		'word/embeddings/Microsoft_Excel_Worksheet.xlsx',
			// 		await fs.readFile(getExcel('04'))
			// 	)
			// 	zip.file(
			// 		'word/embeddings/Microsoft_Excel_Worksheet1.xlsx',
			// 		await fs.readFile(getExcel('08'))
			// 	)
			// }

			await fs.writeFile(
				outputPath,
				zip.generate({
					type: 'nodebuffer',
					compression: 'DEFLATE'
				})
			)
		}
		await fmtTemplate(
			'template_2.docx',
			path.join(tempPath, incrSegment() + '.docx')
		)

		await fs.writeFile(
			path.join(tempPath, incrSegment() + '.docx'),
			await generateSummaryDocx(summary)
		)

		await fmtTemplate(
			'template_3.docx',
			path.join(tempPath, incrSegment() + '.docx')
			// ,true
		)
	}
	let segments = (await fs.readdir(tempPath))
		.sort()
		.map((x) => path.join(tempPath, x))

	segments = [segments[0], segments[2], segments[1]]

	await new Promise<void>((resolve, reject) => {
		const process = spawn(path.join(templatesPath, 'DocxMerge.exe'), [
			'-i',
			...segments,
			'-o',
			path.join(baseDir, Date.now() + '-FullDocumentation.docx')
		])
		process.on('close', (e) => {
			if (process?.exitCode === 0) resolve()
			else reject(e)
		})
		process.on('error', (e) => {
			reject(e)
		})
	})
}
