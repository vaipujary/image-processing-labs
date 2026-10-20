"""Build the editable Lab 5 report from saved results and measured figures.

Run using the Codex bundled document Python or an environment with python-docx
and lxml. Text, tables, and display equations remain native Word objects.
"""
from copy import deepcopy
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
OUTPUT=HERE/'output/docx/Lab_5_Report.docx'


def element(name,children=(),**attrs):
    node=OxmlElement(name)
    for key,value in attrs.items(): node.set(qn(key),str(value))
    for child in children: node.append(deepcopy(child))
    return node


def text(value):
    node=element('m:t'); node.text=value; node.set(qn('xml:space'),'preserve')
    return element('m:r',[node])


def script(base,sub=None,sup=None):
    nodes=[element('m:e',[text(base)] if isinstance(base,str) else base)]
    if sub is not None: nodes.append(element('m:sub',[text(sub)]))
    if sup is not None: nodes.append(element('m:sup',[text(sup)]))
    return element('m:sSubSup' if sub is not None and sup is not None else 'm:sSub' if sub is not None else 'm:sSup',nodes)


def frac(num,den):
    return element('m:f',[element('m:num',num),element('m:den',den)])


def equation(doc,nodes):
    p=doc.add_paragraph()
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(7)
    p._p.append(element('m:oMathPara',[
        element('m:oMathParaPr',[element('m:jc',**{'m:val':'center'})]),element('m:oMath',nodes)]))
    return p


def paragraph(doc,value,style=None):
    return doc.add_paragraph(value,style)


def hyperlink(paragraph,label,url):
    relation=paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True)
    node=element('w:hyperlink',**{'r:id':relation})
    value=element('w:t');value.text=label
    node.append(element('w:r',[element('w:rPr',[element('w:u',**{'w:val':'single'})]),value]))
    paragraph._p.append(node)


def heading(doc,value):
    start_page=bool(doc.paragraphs and doc.paragraphs[-1].text=='_PAGE_BREAK_')
    if start_page:
        node=doc.paragraphs[-1]._p;node.getparent().remove(node)
    p=doc.add_heading(value,level=1)
    p.paragraph_format.page_break_before=start_page
    return p


def figure(doc,name,width,caption=None):
    p=doc.add_paragraph()
    p.paragraph_format.space_after=Pt(2)
    p.paragraph_format.keep_with_next=bool(caption)
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=p.add_run();run.add_picture(str(RESULTS/name),width=Inches(width))
    docpr=run._r.xpath('.//wp:docPr')
    if docpr: docpr[0].set('descr',caption or name.replace('_',' '))
    if caption: paragraph(doc,caption,'Caption')


def page(doc):
    doc.add_paragraph('_PAGE_BREAK_')


def configure(doc):
    for border in doc.styles.element.xpath('.//w:pBdr'): border.getparent().remove(border)
    section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.65);section.bottom_margin=Inches(.65)
    section.left_margin=Inches(.75);section.right_margin=Inches(.75)
    section.footer_distance=Inches(.25)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption','Quote']:
        s=doc.styles[name]
        s.font.name='Arial';s.font.color.rgb=RGBColor(0,0,0)
        s.font.size=Pt(10.5)
        s.paragraph_format.space_after=Pt(6)
    normal=doc.styles['Normal']
    normal.paragraph_format.line_spacing=1.06
    for name,size in [('Title',23),('Subtitle',13),('Heading 1',13),('Heading 2',11),('Caption',9)]:
        doc.styles[name].font.size=Pt(size)
    doc.styles['Title'].font.bold=True
    doc.styles['Heading 1'].font.bold=True
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(9)
    doc.styles['Heading 1'].paragraph_format.space_after=Pt(5)
    doc.styles['Heading 2'].paragraph_format.space_before=Pt(7)
    doc.styles['Heading 2'].paragraph_format.space_after=Pt(3)
    doc.styles['Caption'].font.italic=False
    doc.styles['Caption'].font.bold=False
    doc.styles['Quote'].font.italic=False
    doc.styles['Quote'].font.size=Pt(9.5)
    doc.styles['Quote'].paragraph_format.left_indent=Inches(.18)
    doc.styles['Quote'].paragraph_format.right_indent=Inches(.1)
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=footer.add_run('Lab 5  |  ');r.font.size=Pt(8)
    field=element('w:fldSimple',**{'w:instr':'PAGE'})
    value=element('w:t');value.text='1'
    field.append(element('w:r',[element('w:rPr',[element('w:sz',**{'w:val':'16'})]),value]))
    footer._p.append(field)
    doc.core_properties.title='Lab 5 MAP Reconstruction with Plug and Play'
    doc.core_properties.author='Vaidehi Pujary'
    doc.core_properties.subject='ECE 60141 Lab 5 D1 through D9'


def table(doc, headers, rows, widths):
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for c,value in zip(t.rows[0].cells,headers):c.text=str(value)
    for values in rows:
        for c,value in zip(t.add_row().cells,values):c.text=str(value)
    t._tbl.tblPr.append(element('w:tblBorders',[element('w:'+n,**{'w:val':'single','w:sz':'4','w:color':'D9D9D9'}) for n in ['top','left','bottom','right','insideH','insideV']]))
    for i,row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(element('w:cantSplit'))
        if i==0:row._tr.get_or_add_trPr().append(element('w:tblHeader'))
        for j,c in enumerate(row.cells):
            c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props=c._tc.get_or_add_tcPr()
            props.append(element('w:shd',**{'w:fill':'D9E5F2' if i==0 else 'FFFFFF'}))
            props.append(element('w:tcMar',[element('w:'+side,**{'w:w':'80','w:type':'dxa'}) for side in ['top','left','bottom','right']]))
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(0)
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:r.font.size=Pt(9);r.bold=i==0
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t


def main():
    s=json.loads((RESULTS/'combined_summary.json').read_text());r=s['runs']
    v=json.loads((RESULTS/'verification.json').read_text())
    tests=json.loads((RESULTS/'tests.json').read_text())
    device=json.loads((RESULTS/'dnn_device_check.json').read_text())
    doc=Document();configure(doc)
    paragraph(doc,'Lab 5','Title')
    paragraph(doc,'MAP Reconstruction with Plug and Play','Subtitle')
    paragraph(doc,'ECE 60141 Foundations of Computational Imaging')
    paragraph(doc,'Vaidehi Pujary  •  October 19 2026  •  Random seed 0')
    hyperlink(paragraph(doc,'GitHub repository: '),'vaipujary/image-processing-labs','https://github.com/vaipujary/image-processing-labs')
    paragraph(doc,f'This experiment separates the blur model from the image denoiser. QGGMRF PnP agrees with the converged direct MAP reconstruction to an RMS difference of {s["qggmrf_vs_converged_map_rms"]:.2e}. Replacing the denoiser with DRUNet changes the reconstruction and reduces RMSE from {r["MAP50"]["rmse"]:.5f} for the Lab 4 baseline to {r["D05"]["rmse"]:.5f}.')
    heading(doc,'Experimental setup')
    paragraph(doc,'All runs share the 512 by 768 kodim23.pgm image in [0, 1], circular convolution with Levin kernel 1, and Gaussian noise with σw = 0.02. The seed is 0, and y is bit-for-bit identical to the Lab 4 observation. Image displays use [0, 1]; calculations and RMSEs use unclipped values. Every crop is 128 by 128: rows 160–287 and columns 384–511, zero-based, or [160:288, 384:512].')
    paragraph(doc,'The QGGMRF uses σx = 0.020, p = 2, q = 1.2, T = 1, axial neighbor weights 1/6 and diagonal weights 1/12, excluding pairs outside the image. It runs 100 outer iterations with 20 warm-started inner steps. Each DRUNet experiment runs 50 iterations. FFTs, QGGMRF computations and outer updates use float64 on CPU; the pretrained network uses float32 on Apple MPS, as allowed by the Step 11 wrapper.')
    heading(doc,'D1 Forward model proximal map')
    figure(doc,'D1_forward.png',6.85,'Figure 1. Shared observation and the two forward proximal outputs. Orange boxes identify the common crop.')
    paragraph(doc,f'Task 7a maximum absolute difference between FFT convolution and Lab 3 forward: {s["fft_forward_max_error"]:.3e}. The independent adjoint comparison is {s["fft_adjoint_max_error"]:.3e}.')
    paragraph(doc,'Increasing σp from 0.01 to 1 weakens the penalty for moving away from y, so F(y) makes a much stronger deconvolution correction. The large-σp result amplifies noise in poorly observed frequencies, explaining its noisy appearance despite a smaller data residual.')
    paragraph(doc,f'Data residual RMS ||Ax − y||/√N: y = {s["y_data_residual_rms"]:.6f}; σp = 0.01 gives {r["F_small"]["data_residual_rms"]:.6f}; σp = 1 gives {r["F_large"]["data_residual_rms"]:.6f}.')

    page(doc);heading(doc,'D2 AI specification and corrections')
    paragraph(doc,'The actual request to the assistant was:')
    paragraph(doc,'“can you look at this lab and follow all the instructions to do it in the lab 5 folder I\'ve created? like the previous labs, create a word document for the report so that I can make updates to it once you\'ve populated the report. here is the link: https://cabouman.github.io/grad_labs/labs-60141/lab5/index.html. make no mistakes.”','Quote')
    paragraph(doc,'The assistant translated the linked requirements into the following implementation specification; this is not a separately written student prompt:')
    spec=(HERE/'AI_SPECIFICATION.md').read_text().split('## Assistant implementation specification')[1].split('## Decisions')[0].split('> ',1)[1].strip()
    paragraph(doc,'“'+spec+'”','Quote')
    paragraph(doc,'Corrections and qualifications. The initial quadratic reference test used too few ADMM steps for its weak regularization, so its test prior was strengthened while retaining the independent closed-form reference. The network wrapper separates dtype conversion from CPU/MPS transfers because combined transfers produced invalid values in the installed backend; the invalid trial was discarded and all neural-network runs restarted. The corrected wrapper was checked against CPU before accepting results.')
    paragraph(doc,f'Full-image CPU/MPS denoiser agreement: maximum difference {device["cpu_mps_max_error"]:.2e}, RMS difference {device["cpu_mps_rms_error"]:.2e}. The independent dense linear solve agrees with F to {tests["prox_vs_dense_max_error"]:.2e}, and the QGGMRF gradient agrees with Lab 4 autograd to {tests["prior_gradient_vs_lab4_max_error"]:.2e}.')
    paragraph(doc,'The finite QGGMRF solve remains an approximation; warm starts improve it rather than making it exact. The D4 checklist conflict and the finite-iteration Lab 4 baseline are addressed explicitly below. The full specification and correction record is in AI_SPECIFICATION.md; the implementation is supplied separately as lab5.py.')

    page(doc);heading(doc,'D3 Reproducing the QGGMRF MAP')
    paragraph(doc,'The shared-penalty run uses σp = σn = 0.05 and the same σx = 0.020 prior as Lab 4. The direct solve uses the Lab 4 surrogate steepest-descent update, with equivalent FFT data operations, for 1000 steps; its 50-step checkpoint reproduces the saved Lab 4 R2 image to round-off.')
    figure(doc,'D3_full.png',6.75,'Figure 2. QGGMRF PnP and the converged direct MAP, displayed over [0, 1].')
    figure(doc,'D3_crops.png',6.75,'Figure 3. Shared crop [160:288, 384:512], including the original 50-step Lab 4 baseline.')
    table(doc,['Reconstruction','Full image RMSE'],[
        ['QGGMRF PnP, 100 × 20 steps',f'{r["Q05"]["rmse"]:.8f}'],
        ['Direct MAP, 1000 steps',f'{r["MAP_converged"]["rmse"]:.8f}'],
        ['Lab 4 R2, 50 steps',f'{r["MAP50"]["rmse"]:.8f}']],[4.5,2.1])
    paragraph(doc,f'PnP reproduces the MAP to an RMS difference of {s["qggmrf_vs_converged_map_rms"]:.3e}; its RMS difference from the earlier 50-step baseline is {s["qggmrf_vs_map50_rms"]:.8f}, reflecting incomplete convergence of that baseline.')
    paragraph(doc,f'The direct MAP gradient RMS is {v["runs"]["MAP_converged"]["weighted_map_gradient_rms"]:.2e}; QGGMRF PnP ends at relative consensus residual {r["Q05"]["history"][-1]:.2e}. No clipping or postprocessing is used to improve agreement.')

    page(doc);heading(doc,'D4 QGGMRF regularization sweep')
    paragraph(doc,'This sweep fixes σn = 0.05 and varies only σp, with all other settings unchanged. The warm-started inner solver uses 20 steps in each of 100 outer iterations.')
    equation(doc,[text('∇f(x*) + (σn²/σp²) ∇h(x*) = 0')])
    figure(doc,'D4_sweep.png',6.85,'Figure 4. QGGMRF sweep in the common 128 by 128 crop; display range [0, 1].')
    table(doc,['σp','σn² / σp²','Full image RMSE'],[[f'{sp:.3f}',f'{(.05/sp)**2:g}',f'{r[k]["rmse"]:.8f}'] for k,sp in [('Q025',.025),('Q05',.05),('Q10',.1)]],[1.5,2.2,2.9])
    paragraph(doc,'A smaller σp raises the prior weight σn²/σp² and produces stronger smoothing, whereas a larger σp lowers the weight and preserves sharper transitions together with more noise.')
    paragraph(doc,'Instruction discrepancy. Task 10b and the experiment table hold σn fixed, yielding weights 4, 1 and 0.25 and different RMSEs. The final checklist asks why the sweep RMSEs are the same, which would instead describe tying σn to σp and converging to one fixed objective. The results here follow the explicit Task 10b settings and its weighted-MAP equation.')
    heading(doc,'Accuracy of the inner solve')
    figure(doc,'Q_inner_comparison.png',6.3,'Figure 5. Additional check for D9 question 2: the same shared-penalty problem with 20 or 10 inner steps. Both histories use a log vertical axis.')

    page(doc);heading(doc,'D5 Deblurring with DRUNet')
    paragraph(doc,'DRUNet PnP ADMM uses σp = 0.05, σn = 0.03 and 50 iterations on the same y. The comparison baseline is the original 50-step Lab 4 MAP, matching its reported result.')
    figure(doc,'D5_full.png',6.85,'Figure 6. Observation, DRUNet PnP and Lab 4 MAP, over [0, 1].')
    figure(doc,'D5_crops.png',6.85,'Figure 7. Ground truth and all three estimates in [160:288, 384:512], over [0, 1].')
    table(doc,['Image','Full image RMSE'],[['Blurred noisy y',f'{s["input_rmse"]:.8f}'],['DRUNet PnP ADMM',f'{r["D05"]["rmse"]:.8f}'],['Lab 4 MAP, 50 steps',f'{r["MAP50"]["rmse"]:.8f}']],[4.5,2.1])
    figure(doc,'D5_convergence.png',6.1,'Figure 8. Relative consensus residual ||x − v||/||x|| versus outer iteration, log vertical axis.')
    paragraph(doc,f'DRUNet PnP reduces RMSE by {100*(1-r["D05"]["rmse"]/r["MAP50"]["rmse"]):.1f}% relative to Lab 4 MAP, with a cleaner crop and clearer edge detail; its final consensus residual is {r["D05"]["history"][-1]:.3e}.')

    page(doc);heading(doc,'D6 DRUNet regularization sweep')
    paragraph(doc,'The denoiser noise level remains σn = 0.03; σp is halved and doubled around the D5 value. Each result uses the same observation and exactly 50 iterations.')
    figure(doc,'D6_sweep.png',6.85,'Figure 9. DRUNet PnP crops for σp = 0.025, 0.05 and 0.10; display range [0, 1].')
    table(doc,['σp','σn','Full image RMSE'],[[f'{sp:.3f}','0.030',f'{r[k]["rmse"]:.8f}'] for k,sp in [('D025',.025),('D05',.05),('D10',.1)]],[1.5,1.5,3.6])
    paragraph(doc,'With σn fixed, smaller σp makes the forward update weaker relative to denoising and over-regularizes, smoothing fine structure. Larger σp gives the forward update more influence and under-regularizes, allowing more noise and artifacts; for DRUNet this describes an empirical balance, not a proven explicit prior weight.')
    heading(doc,'D7 Optional unstable proximal gradient run')
    figure(doc,'D7_unstable.png',6.5,'Figure 10. PnP PG at σp = 0.05 and σn = 0.03. The blue curve is the required normalized iterate change; the dotted red curve tracks image RMS on its own log scale.')
    paragraph(doc,f'The explicit data-gradient step violates σp ≤ √2 σw = {v["stability_ceiling"]:.8f}, since 0.05 exceeds that bound.')
    paragraph(doc,f'The image RMS grows from {r["PG_unstable"]["output_rms_history"][0]:.3g} to {r["PG_unstable"]["output_rms_history"][-1]:.3g}; the normalized change ends at {r["PG_unstable"]["history"][-1]:.3g}. A relative change can level off while both its numerator and denominator grow, so this nonzero plateau is not evidence of convergence. DRUNet nonexpansiveness is not established; the bound characterizes the linear gradient step, not a global guarantee for the network.')

    page(doc);heading(doc,'D8 Optional stable proximal gradient run')
    paragraph(doc,'Halving both parameters gives σp = 0.025 and σn = 0.015, keeping σn²/σp² = 0.36 while satisfying the explicit-gradient bound. Both methods below run for 50 iterations.')
    figure(doc,'D8_full.png',5.8,'Figure 11. Rescaled PnP PG and the D5 PnP ADMM result, over [0, 1].')
    figure(doc,'D8_crops.png',4.5,'Figure 12. The same comparison in crop [160:288, 384:512], over [0, 1].')
    table(doc,['Algorithm','Full image RMSE'],[['PnP PG, rescaled',f'{r["PG_stable"]["rmse"]:.8f}'],['PnP ADMM, D5',f'{r["D05"]["rmse"]:.8f}']],[4.5,2.1])
    figure(doc,'D8_convergence.png',5.5,'Figure 13. PnP PG iterate change and PnP ADMM consensus residual on one log vertical axis; these are distinct diagnostics.')
    settle=v['below_1e_3_through_end']
    phrase=lambda key: str(settle[key]) if settle[key] is not None else 'not reached in 50 steps'
    paragraph(doc,f'ADMM preserves more beak detail and has lower RMSE ({r["D05"]["rmse"]:.5f} versus {r["PG_stable"]["rmse"]:.5f}); its consensus residual stays below 10⁻³ from iteration {phrase("D05")}, while PG’s relative iterate change reaches that criterion at iteration {phrase("PG_stable")}.')
    paragraph(doc,'Equal parameter ratios imply the same ideal weighted MAP target for exact proximal denoisers, but changing the DRUNet noise level changes the learned operator; identical images are therefore not guaranteed.')

    page(doc);heading(doc,'D9 Answers to the five questions')
    doc.add_heading('1 Settings needed to reproduce MAP',level=2)
    paragraph(doc,f'I used one shared penalty σn = σp = 0.05, the unchanged prior scale σx = 0.020 with p = 2, q = 1.2 and T = 1, and warm starts from the previous denoiser output. These settings give an RMS difference of {s["qggmrf_vs_converged_map_rms"]:.2e} from the converged direct MAP. The equality σn = σp is the single coupling I relax for the general two-knob PnP setting: D5 uses 0.03 and 0.05. Replacing the QGGMRF agent with DRUNet also removes the explicit prior and the need for an iterative warm start.')
    doc.add_heading('2 Halving the inner iteration count',level=2)
    paragraph(doc,f'Fewer inner steps make each approximate proximal evaluation less accurate, so I would expect a slower residual decrease, possible extra fluctuations, or a higher finite-iteration residual. Warm starts can make the difference small because the next denoising solution is close to the previous one. The additional 10-step experiment confirms that here: the final residual is {r["Q05_inner10"]["history"][-1]:.2e}, versus {r["Q05"]["history"][-1]:.2e} with 20 steps, and the RMSEs agree to seven decimal places. A persistent residual floor is possible with insufficient solves but is not forced by halving this budget.')
    doc.add_heading('3 Where neural regularization strength comes from',level=2)
    paragraph(doc,f'The two knobs are the forward-prox scale σp and the denoiser noise level σn. A higher σn requests stronger denoising; at fixed σn, a smaller σp weakens each data correction relative to denoising. At σn = 0.03, the sweep RMSEs are {r["D025"]["rmse"]:.5f}, {r["D05"]["rmse"]:.5f} and {r["D10"]["rmse"]:.5f}. I therefore tune both parameters using image detail, artifacts and a held-out reference when available, and separately check the residual; σx is not a DRUNet parameter.')
    doc.add_heading('4 A denoiser that returns only the image mean',level=2)
    paragraph(doc,'Let P project an image onto the constant-image subspace. The loop becomes x = F(v − u), v = P(x + u), and u ← u + x − v. After every denoising step v is constant, and the dual update removes the mean of u. At consensus x is also constant; because this normalized circular blur maps a constant c to itself, minimizing the data term over constants gives c = mean(y).')
    equation(doc,[text('x* = mean(y) 1')])
    paragraph(doc,f'The mean-only check approaches mean(y) = {v["mean_denoiser"]["target_mean"]:.6f}, with final residual {v["mean_denoiser"]["final_residual"]:.2e} and RMSE {v["mean_denoiser"]["rmse"]:.5f}. The parrot and all spatial detail disappear even though the two agents agree. Consensus tests agreement between the chosen agents, not whether those agents preserve the desired image.')
    doc.add_heading('5 A third consensus equilibrium agent',level=2)
    paragraph(doc,'I would add an agent enforcing independently calibrated intensity bounds for the latent image. F encodes the blur and measurement noise, while DRUNet supplies learned image statistics; neither strictly enforces those physical bounds in this experiment. The proximal map of the indicator of [0, 1]ᴺ projects each pixel onto [0, 1], leaving feasible values unchanged and moving out-of-range values to the nearest bound. Such an agent would be incorporated into a three-agent consensus formulation, rather than added as an undocumented clipping step to these runs.')

    page(doc);heading(doc,'Reproducibility and sources')
    paragraph(doc,f'The numerical environment used PyTorch {s["torch"]}, DeepInv {s["deepinv"]}, four CPU threads and float32 MPS inference. The pretrained grayscale checkpoint is drunet_deepinv_gray_finetune_26k.pth; its SHA256 is recorded in results/verification.json. All measured arrays, histories, input hashes and numerical checks are saved under lab5/results. All required function signatures and the functions-only file structure were checked automatically.')
    paragraph(doc,'The report contains D1 through D9 in order, including both optional experiments. Text, tables and equations are editable Word objects. Figures are replaceable images generated from the saved arrays; code is supplied as lab5.py rather than pasted into the report. The matching PDF is the version intended for the lab submission.')
    hyperlink(paragraph(doc,'Assignment: '),'Lab 5 MAP Reconstruction with Plug and Play','https://cabouman.github.io/grad_labs/labs-60141/lab5/index.html')
    paragraph(doc,'The assignment snapshot used here is saved in references/lab5_instructions.html, accessed October 19 2026. The report follows the explicit experiment settings where the final D4 checklist conflicts with Task 10b.')
    hyperlink(paragraph(doc,'Network documentation: '),'DeepInv DRUNet API','https://deepinv.org/api/stubs/deepinv.models.DRUNet.html')
    hyperlink(paragraph(doc,'Pretrained checkpoint source: '),'DeepInv DRUNet model repository','https://huggingface.co/deepinv/drunet')
    paragraph(doc,'The same local Lab 3 image and kernel and the Lab 4 R2 results provide the experimental inputs and comparison. No reference-image pixels are used inside either reconstruction algorithm; the clean image is used only for evaluation and figures.')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);doc.save(OUTPUT)
    print(OUTPUT)


if __name__=='__main__': main()
