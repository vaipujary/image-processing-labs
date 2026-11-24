"""Create the editable Lab 6 report from measured results.

Run with the bundled document Python. Equations, text, and tables are native
Word objects. Figures are embedded PNGs, with editable source in make_figures.py.
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
OUT=HERE/'output/Lab_6_Report.docx'
RESULTS=HERE/'results'

def el(name,children=(),**attrs):
    n=OxmlElement(name)
    for k,v in attrs.items():n.set(qn(k),str(v))
    for c in children:n.append(deepcopy(c))
    return n

def t(value):
    node=el('m:t');node.text=str(value);node.set(qn('xml:space'),'preserve')
    return el('m:r',[node])

def nodes(x):return [t(x)] if isinstance(x,(str,int,float)) else x if isinstance(x,list) else [x]

def sub(base,index):return el('m:sSub',[el('m:e',nodes(base)),el('m:sub',nodes(index))])
def sup(base,index):return el('m:sSup',[el('m:e',nodes(base)),el('m:sup',nodes(index))])
def frac(a,b):return el('m:f',[el('m:num',nodes(a)),el('m:den',nodes(b))])
def bar(a):return el('m:acc',[el('m:accPr',[el('m:chr',**{'m:val':'̄'})]),el('m:e',nodes(a))])
def op(symbol,lower,upper,body):
    props=[el('m:chr',**{'m:val':symbol}),el('m:limLoc',**{'m:val':'subSup'})]
    if upper is None:props.append(el('m:supHide',**{'m:val':'1'}))
    return el('m:nary',[el('m:naryPr',props),el('m:sub',nodes(lower)),el('m:sup',nodes(upper or '')),el('m:e',nodes(body))])

def matrix(rows):
    return el('m:d',[el('m:dPr',[el('m:begChr',**{'m:val':'['}),el('m:endChr',**{'m:val':']'})]),el('m:e',[
        el('m:m',[el('m:mPr',[el('m:cGpRule',**{'m:val':'4'}),el('m:cGp',**{'m:val':'120'}),
                              el('m:mcs',[el('m:mc',[el('m:mcPr',[el('m:count',**{'m:val':str(len(rows[0]))}),el('m:mcJc',**{'m:val':'center'})])])])])]+
                   [el('m:mr',[el('m:e',[t(v)]) for v in row]) for row in rows])])])

def mathparagraph(parent,items):
    p=parent.add_paragraph() if hasattr(parent,'add_paragraph') else parent
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(6)
    p._p.append(el('m:oMathPara',[el('m:oMathParaPr',[el('m:jc',**{'m:val':'center'})]),el('m:oMath',items)]))
    return p

def para(doc,value,style=None):return doc.add_paragraph(value,style)

def link(p,label,url):
    rid=p.part.relate_to(url,RT.HYPERLINK,is_external=True)
    tx=el('w:t');tx.text=label
    p._p.append(el('w:hyperlink',[el('w:r',[el('w:rPr',[el('w:u',**{'w:val':'single'})]),tx])],**{'r:id':rid}))

def heading(doc,value,level=1,newpage=False):
    p=doc.add_heading(value,level=level)
    if newpage:p.paragraph_format.page_break_before=True
    return p

def figure(doc,name,width,caption):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.keep_with_next=True
    r=p.add_run();r.add_picture(str(RESULTS/name),width=Inches(width))
    r._r.xpath('.//wp:docPr')[0].set('descr',caption)
    para(doc,caption,'Caption')

def table(doc,headers,rows,widths):
    tab=doc.add_table(rows=1,cols=len(headers));tab.autofit=False
    tab.alignment=WD_TABLE_ALIGNMENT.CENTER
    for c,w in zip(tab.columns,widths):c.width=Inches(w)
    for c,value in zip(tab.rows[0].cells,headers):c.text=value
    for row in rows:
        for c,value in zip(tab.add_row().cells,row):
            if isinstance(value,list):
                c.paragraphs[0]._p.append(el('m:oMath',[matrix(value)]))
            else:c.text=str(value)
    tab._tbl.tblPr.append(el('w:tblBorders',[el('w:'+side,**{'w:val':'single','w:sz':'4','w:color':'D9D9D9'}) for side in ['top','left','bottom','right','insideH','insideV']]))
    for i,row in enumerate(tab.rows):
        row._tr.get_or_add_trPr().append(el('w:cantSplit'))
        if i==0:row._tr.get_or_add_trPr().append(el('w:tblHeader'))
        for j,c in enumerate(row.cells):
            c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props=c._tc.get_or_add_tcPr()
            props.append(el('w:shd',**{'w:fill':'DCE6F0' if i==0 else 'FFFFFF'}))
            props.append(el('w:tcMar',[el('w:'+side,**{'w:w':'85','w:type':'dxa'}) for side in ['top','left','bottom','right']]))
            for p in c.paragraphs:
                p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
                p.paragraph_format.line_spacing=1.02
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:r.font.size=Pt(9);r.bold=i==0
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
    p.paragraph_format.line_spacing=Pt(4);p.add_run().font.size=Pt(4)
    return tab

def configure(doc):
    section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.65);section.bottom_margin=Inches(.65)
    section.left_margin=Inches(.75);section.right_margin=Inches(.75)
    section.footer_distance=Inches(.28)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
        s=doc.styles[name];s.font.name='Arial';s.font.size=Pt(10.5);s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.space_after=Pt(6)
    doc.styles['Normal'].paragraph_format.line_spacing=1.06
    for name,size in [('Title',23),('Subtitle',14),('Heading 1',14),('Heading 2',11),('Heading 3',10.5),('Caption',9)]:
        doc.styles[name].font.size=Pt(size)
    for name in ['Title','Heading 1','Heading 2','Heading 3']:doc.styles[name].font.bold=True
    for name in ['Heading 1','Heading 2','Heading 3']:
        doc.styles[name].paragraph_format.space_before=Pt(10)
        doc.styles[name].paragraph_format.space_after=Pt(5)
    doc.styles['Caption'].font.italic=False
    for border in doc.styles.element.xpath('.//w:pBdr'):border.getparent().remove(border)
    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('Lab 6  |  ').font.size=Pt(8)
    tx=el('w:t');tx.text='1'
    footer._p.append(el('w:fldSimple',[el('w:r',[tx])],**{'w:instr':'PAGE'}))
    doc.core_properties.title='Lab 6 The EM Algorithm'
    doc.core_properties.author='Vaidehi Pujary'
    doc.core_properties.subject='ECE 60141 Gaussian mixture estimation and MDL model order selection'

def spec(doc,name,specs):
    s=specs[name]
    heading(doc,{'e_step':'Posterior probabilities','m_step':'Parameter update','log_likelihood':'Observed data log likelihood','run_em':'EM iteration loop','mdl':'MDL score','merge_closest':'Closest component merge','estimate_order':'Model order search'}[name],2)
    p=para(doc,s['signature']);p.paragraph_format.keep_with_next=True
    for r in p.runs:r.font.name='Courier New';r.font.size=Pt(9)
    for label,key in [('Purpose','purpose'),('Arguments','arguments'),('Returns','returns'),('Invariants','invariants')]:
        p=doc.add_paragraph();p.add_run(label+'. ').bold=True;p.add_run(s[key])

def mat(R):return [[f'{v:.4f}' for v in row] for row in R]
def vec(v):return [[f'{x:.4f}'] for x in v]
def vector(v,n=4):return '('+', '.join(f'{x:.{n}f}' for x in v)+')'

def main():
    s=json.loads((RESULTS/'summary.json').read_text());v=json.loads((RESULTS/'verification.json').read_text())
    specs=json.loads((HERE/'SPECIFICATIONS.json').read_text())
    runs=s['runs'];b=runs['baseline'];c=s['checks'];truth=s['truth'];order=s['order']
    doc=Document();configure(doc)
    para(doc,'Lab 6','Title');para(doc,'The EM Algorithm','Subtitle')
    para(doc,'ECE 60141 Foundations of Computational Imaging')
    para(doc,'Vaidehi Pujary  •  November 15 2026')
    link(para(doc,'GitHub repository: '),'vaipujary/image-processing-labs','https://github.com/vaipujary/image-processing-labs')
    para(doc,'Random generator: NumPy default_rng with PCG64. Data seed 0; random-start seeds 0–4. Experiments use the labs conda environment, Python '+s['environment']['python']+' and NumPy '+s['environment']['numpy']+'.')
    para(doc,f'This experiment estimates a Gaussian mixture with EM and selects its order by merging components and minimizing MDL. The search selects three components, the true order, with cost {s["merge3"]["cost"]:.6f}. Initialization matters: the prescribed baseline has cost {b["cost"]:.6f} and {b["error_fraction"]:.1%} label disagreement after 20 iterations.')
    p=para(doc,'Editing note. This is an AI-assisted working draft. D2 contains reference derivations; the lab requires your own unaided derivations. D3 and D10 contain assistant-written specifications, D8 identifies the assistant’s recorded prediction, and D13 question 6 describes the actual build history. Review these sections before submission.');p.runs[0].italic=True;p.runs[0].font.size=Pt(9)
    heading(doc,'D1 Generated data')
    para(doc,'The 500 two-dimensional observations were generated once and saved with their true labels in results/data.npz. Each label draw is immediately followed by its Gaussian sample draw; subsequent experiments load this same file. The class counts are 187, 198, and 115 for classes 0, 1, and 2, respectively.')
    figure(doc,'D1_true_data.png',6.1,'Figure 1. True class labels for all 500 observations. Equal axis scaling preserves covariance geometry.')

    heading(doc,'D2 Reference derivations',newpage=True)
    para(doc,'These derivations are supplied as AI-assisted reference material. Replace them with your independently completed work to satisfy Step 5.')
    heading(doc,'Posterior component probability',2)
    para(doc,'Write fₘ(yₙ) for the component Gaussian density and gᵢ for the mixture density at observation i. Independence of the observation–label pairs gives the joint density below; all observations except n contribute the same factors in the numerator and denominator.')
    mathparagraph(doc,[sub('g','i'),t(' = '),op('∑','j = 0','M − 1',[sub('π','j'),sub('f','j'),t('('),sub('y','i'),t(')')])])
    mathparagraph(doc,[t('p('),sub('X','n'),t(' = m, Y = y | θ) = '),sub('π','m'),sub('f','m'),t('('),sub('y','n'),t(') '),op('∏','i ≠ n',None,sub('g','i'))])
    mathparagraph(doc,[t('p(y | θ) = '),sub('g','n'),t(' '),op('∏','i ≠ n',None,sub('g','i'))])
    para(doc,'Bayes’ rule cancels those common factors. Thus the posterior depends on the full data only through yₙ, giving equation (6):')
    mathparagraph(doc,[t('P{'),sub('X','n'),t(' = m | Y = y, θ} = '),frac([sub('π','m'),sub('f','m'),t('('),sub('y','n'),t(') '),op('∏','i ≠ n',None,sub('g','i'))],[sub('g','n'),t(' '),op('∏','i ≠ n',None,sub('g','i'))]),t(' = '),frac([sub('π','m'),sub('f','m'),t('('),sub('y','n'),t(')')],op('∑','j = 0','M − 1',[sub('π','j'),sub('f','j'),t('('),sub('y','n'),t(')')]))])
    heading(doc,'Hard assignments recover labeled maximum likelihood',2)
    para(doc,'For known labels, set Pₙ,ₘ = δ(xₙ − m). Substitution into equation (7) gives the sufficient statistics of equation (4) exactly:')
    mathparagraph(doc,[sub(bar('N'),'m'),t(' = '),op('∑','n = 1','N',[t('δ('),sub('x','n'),t(' − m)')]),t(' = '),sub('N','m')])
    mathparagraph(doc,[sub(bar('b'),'m'),t(' = '),op('∑','n = 1','N',[sub('y','n'),t(' δ('),sub('x','n'),t(' − m)')]),t(' = '),sub('b','m')])
    mathparagraph(doc,[sub(bar('S'),'m'),t(' = '),op('∑','n = 1','N',[sub('y','n'),sup(sub('y','n'),'T'),t(' δ('),sub('x','n'),t(' − m)')]),t(' = '),sub('S','m')])
    para(doc,'Equation (8) therefore reduces to equation (5):')
    mathparagraph(doc,[sub('π','m'),t(' = '),frac(sub('N','m'),'N'),t(',   '),sub('μ','m'),t(' = '),frac(sub('b','m'),sub('N','m')),t(',   '),sub('R','m'),t(' = '),frac(sub('S','m'),sub('N','m')),t(' − '),frac([sub('b','m'),sup(sub('b','m'),'T')],sup(sub('N','m'),'2'))])
    para(doc,'This is the ML covariance with divisor Nₘ. The implementation adds 10⁻⁶I afterward as Step 7 requires; D4 adds the identical term to the direct covariance before comparing.')

    heading(doc,'D3 Specifications for the four EM functions',newpage=True)
    para(doc,'The following specifications were written before implementation and are reproduced verbatim from SPECIFICATIONS.json. Each function was checked before proceeding to the next function; the complete numerical results appear in D4.')
    spec(doc,'e_step',specs);spec(doc,'m_step',specs)
    heading(doc,'D3 Likelihood and iteration specifications',newpage=True)
    spec(doc,'log_likelihood',specs);spec(doc,'run_em',specs)
    para(doc,'Implementation convention. The initial likelihood is history[0]. D5 and D8 use history[1:] so their x axes show exactly iterations 1 through 20. Calculations use float64; all logarithms are natural.')

    heading(doc,'D4 Numerical checks',newpage=True)
    table(doc,['Check','Measured result','Outcome'],[
        ['Posterior row normalization',f'Maximum row-sum error {c["e_step_max_row_error"]:.3e}','Pass'],
        ['Distant point at (1000, 1000)','Posterior (1, 0, 0); row sum 1; all entries finite','Pass'],
        ['Hard assignments against labeled ML',f'Max |Δπ| = {c["hard_errors"][0]:.3e}\nMax |Δμ| = {c["hard_errors"][1]:.3e}\nMax |ΔR| = {c["hard_errors"][2]:.3e}','Pass'],
        ['Single Gaussian likelihood',f'Implementation {c["single_gaussian_ll"]:.12f}\nDirect density {c["direct_gaussian_ll"]:.12f}\nAbsolute difference {c["gaussian_error"]:.3e}','Pass'],
        ['EM likelihood monotonicity',f'Most negative change {c["largest_decrease"]:.3e}\nAcceptance tolerance −1e−8','Pass']],[2.05,3.95,0.8])
    para(doc,'The distant point makes all three directly evaluated Gaussian densities equal to floating-point zero. Normalizing those values would divide zero by zero; the log-domain E-step remains finite. The hard-assignment check uses class fractions (0.374, 0.396, 0.230), class sample means, and covariances with the ML divisor and the required ridge.')
    para(doc,'For the single-Gaussian check, the mean and ML covariance were estimated from all 500 samples, and 10⁻⁶I was added to the covariance. The independent comparison sums scipy.stats.multivariate_normal.logpdf. Monotonicity was checked across all 16 EM histories: six three-component fits, the true-parameter initialization, and nine orders. The tiny decrease occurs at M = 2 and is rounding-level; merge jumps are excluded because they change the model order.')
    heading(doc,'Initialization at the true parameters',2)
    table(doc,['True class','Fitted weight','Fitted mean','Mean error norm','Covariance error norm'],[
        [str(i),f'{c["true_start_weights"][i]:.6f}',vector(c['true_start_means'][i]),f'{c["true_start_mean_errors"][i]:.6f}',f'{c["true_start_cov_errors"][i]:.6f}'] for i in range(3)],[.7,1.05,2.35,1.25,1.45])
    para(doc,f'After 20 iterations from the true parameters, the largest weight error is {c["true_start_weight_error"]:.6f}, the largest Euclidean mean error is {max(c["true_start_mean_errors"]):.6f}, and the largest absolute covariance-entry error is {c["true_start_cov_entry_error"]:.6f}. The table uses Frobenius covariance error norms. These are modest finite-sample changes: the generating parameters need not maximize the likelihood of this particular sample.')
    para(doc,f'Additional audit. A separate implementation using explicit two-dimensional Gaussian formulas and raw second moments reproduces all six fits within {max(v["independent_em_max_errors"].values()):.3e}. Independent determinant calculations select all eight of the same merge pairs. The merge preserves the mean and second raw moment to the displayed precision; both maximum errors are {max(c["merge_moment_errors"]):.1e}.')

    heading(doc,'D5 Cost for the prescribed initialization',newpage=True)
    para(doc,'The three initial means are the first three saved observations. All initial weights are 1/3 and every initial covariance is the identity. Exactly 20 EM iterations are used.')
    figure(doc,'D5_baseline_cost.png',6.7,'Figure 2. Negative log likelihood after each of the 20 EM iterations. Iteration zero is excluded.')
    para(doc,f'Yes, the cost is still decreasing at iteration 20. It changes from {-b["history"][-2]:.9f} at iteration 19 to {b["cost"]:.9f} at iteration 20, a decrease of {b["history"][-1]-b["history"][-2]:.9f}. The initial cost, before any EM updates, is {-b["history"][0]:.9f}.')
    heading(doc,'Observed initialization effect',2)
    table(doc,['Initial component','Sample index','Initial mean','True class'],[[str(i),str(i),vector(b['initial_means'][i]),str(b['initial_true_labels'][i])] for i in range(3)],[1.25,1.15,3.2,1.2])
    para(doc,'Two initial means lie in the negative cluster, and no initial mean lies in the rightmost positive cluster. After 20 iterations the fit still splits the negative group and combines the two positive groups. The lab’s description anticipates a reasonable baseline fit; the specified interleaved draw order produces a poorer fit here. The independent numerical audit confirms this result, and the data or initialization were not changed to obtain a preferred outcome.')

    heading(doc,'D6 Estimated and true parameters',newpage=True)
    para(doc,'The correspondence minimizes the sum of Euclidean mean distances over all six permutations. The selected assignment is true class 0 → estimated component 0, true class 1 → estimated component 2, and true class 2 → estimated component 1. All values below use this single one-to-one matching.')
    rows=[]
    for i in range(3):
        rows.extend([[f'Class {i}', 'Weight',f'{truth["weights"][i]:.4f}',f'{b["matched_weights"][i]:.4f}'],
                     ['', 'Mean',vec(truth['means'][i]),vec(b['matched_means'][i])],
                     ['', 'Covariance',mat(truth['covariances'][i]),mat(b['matched_covariances'][i])]])
    table(doc,['True class','Parameter','True value','Estimated value'],rows,[.75,1.05,2.5,2.5])
    para(doc,'The large discrepancies are meaningful. The positive component has weight 0.6035 and variance 3.9484 in the first coordinate because it covers both positive clusters. The other two fitted means remain near the negative cluster. One-to-one matching assigns one of these negative components to true class 0 even though it is far from that class; matching cannot repair a poor fit.')
    para(doc,'Covariances are symmetric matrices including the required 10⁻⁶I term. This table reports the requested 20-iteration baseline, rather than substituting the better model obtained by the later merge search.')

    heading(doc,'D7 Hard labels and disagreement',newpage=True)
    para(doc,'Posterior probabilities are recomputed using the final parameters. Each point is assigned to the component with the largest posterior probability, including its fitted mixture weight. The permutation from D6 is then used to compare labels with the true classes.')
    figure(doc,'D7_baseline_labels.png',6.7,'Figure 3. Baseline hard assignments after mean matching. X markers show fitted means; both axes use the same scale.')
    para(doc,f'The disagreement fraction is {b["errors"]}/500 = {b["error_fraction"]:.3f}, or {b["error_fraction"]:.1%}. This large value reflects the split negative cluster and combined positive clusters, as well as the required one-to-one mean matching.')
    table(doc,['True class','Assigned class 0','Assigned class 1','Assigned class 2'],[[str(i),*row] for i,row in enumerate(b['confusion'])],[1.2,1.85,1.85,1.9])
    para(doc,'The table confirms that 186 of the 187 true class-0 points are assigned to the component matched to class 2. True class 1 is split mainly between components matched to classes 0 and 1.')

    heading(doc,'D8 Five random initializations',newpage=True)
    para(doc,'Prediction recorded before the runs, by the assistant: the five random starts will not all reach the same final cost as the first-three-samples start after 20 iterations, because the starting means can lead to different likelihood basins or convergence rates.')
    figure(doc,'D8_all_costs.png',6.6,'Figure 4. The five seeded starts and the prescribed baseline, each run for exactly 20 iterations.')
    table(doc,['Initialization','Initial sample indices','Final cost','Disagreement'],[
        ['First three samples',', '.join(map(str,b['initial_indices'])),f'{b["cost"]:.6f}',f'{b["error_fraction"]:.1%}']]+
        [[f'Seed {i}',', '.join(map(str,runs[f'seed_{i}']['initial_indices'])),f'{runs[f"seed_{i}"]["cost"]:.6f}',f'{runs[f"seed_{i}"]["error_fraction"]:.1%}'] for i in range(5)],[1.6,2.1,1.65,1.45])
    random_costs=[runs[f'seed_{i}']['cost'] for i in range(5)]
    para(doc,f'The recorded prediction is supported. The random-start costs range from {min(random_costs):.6f} to {max(random_costs):.6f}, a spread of {max(random_costs)-min(random_costs):.6f}. Seed 3 is best; seed 2 remains in a poor clustering. Seeds 0, 1, and 4 recover the main cluster regions but have different residual optimization errors after the fixed iteration budget.')

    heading(doc,'D9 Random start cluster plots',newpage=True)
    para(doc,'Seeds 0, 1, 3, and 4 recover the three main groups. Seed 2 uses two components for the negative group and one component for both positive groups; its final cost is 1907.424010.')
    figure(doc,'D9_five_starts.png',6.6,'Figure 5. Hard-label scatter plots for all five random starts, with fitted means marked and each final cost shown. Colors match D1 after the same one-to-one mean matching used for D7.')

    heading(doc,'D10 Specifications for model order estimation',newpage=True)
    para(doc,'These three specifications were written before their implementation and are reproduced verbatim from SPECIFICATIONS.json.')
    spec(doc,'mdl',specs);spec(doc,'merge_closest',specs)
    heading(doc,'D10 Search loop specification',newpage=True)
    spec(doc,'estimate_order',specs)
    heading(doc,'Search configuration',2)
    para(doc,'The search starts with nine components, weights 1/9, identity covariances, and the first nine saved observations as means. Each order receives exactly 20 EM iterations. The search performs 180 EM updates and eight moment-preserving merges, saving each fitted order before continuing downward.')
    para(doc,'For p = 2, each component contributes one weight, two mean entries, and three distinct covariance entries; the sum-to-one constraint removes one weight parameter. Consequently L(M) = 6M − 1. The penalty uses log(500 × 2) = log(1000), as required, rather than log(500).')
    mathparagraph(doc,[t('MDL(M, θ) = −log p(y | M, θ) + '),frac('6M − 1','2'),t(' log(1000)')])

    heading(doc,'D11 MDL throughout the merge search',newpage=True)
    figure(doc,'D11_mdl_trace.png',6.8,'Figure 6. MDL after all 180 EM updates and immediately after each of the eight merges. A merge adds a second point at the same cumulative iteration, shown by a diamond and vertical dotted line.')
    records=[]
    for event in order['merges']:
        it=event['iteration'];M=event['from_order']
        pre=next(q['mdl'] for q in order['trace'] if q['iteration']==it and q['event']=='em')
        post=next(q['mdl'] for q in order['trace'] if q['iteration']==it and q['event']=='merge')
        records.append([str(it),f'{M} → {M-1}',f'{pre:.6f}',f'{post:.6f}',f'{post-pre:+.6f}'])
    table(doc,['Iteration','Merge','MDL before','MDL after','Change'],records,[.85,.95,1.7,1.7,1.6])
    para(doc,'Within each fixed order, the penalty is constant and EM lowers MDL up to numerical precision. At a merge, both the likelihood and the penalty change, so the curve can jump either downward or upward. Reducing nine components to three decreases MDL overall. The merges to two and one components increase it, indicating that their loss of fit outweighs the saved parameters.')

    heading(doc,'D12 Selected number of components',newpage=True)
    figure(doc,'D12_mdl_order.png',6.6,'Figure 7. Final MDL at each order after its 20 EM iterations. The minimum occurs at M = 3.')
    records=[]
    for M in range(1,10):
        model=order['models'][str(M)];cost=-model['history'][-1]
        records.append([str(M),str(6*M-1),f'{cost:.6f}',f'{model["mdl"]-cost:.6f}',f'{model["mdl"]:.6f}'])
    table(doc,['M','L(M)','Cost','Penalty','MDL'],records,[.55,.65,1.85,1.8,1.95])
    para(doc,f'The minimizing order is M̂ = {order["best_order"]}, which equals the true value M = 3. Its MDL is {order["models"]["3"]["mdl"]:.6f}. The next lowest score is {order["models"]["4"]["mdl"]:.6f} at M = 4, a difference of {order["models"]["4"]["mdl"]-order["models"]["3"]["mdl"]:.6f}. All higher-order fits improve likelihood relative to M = 3, but none improves the penalized criterion.')
    para(doc,'The measured cost is not monotone across all orders: M = 8 and M = 9 have higher costs than M = 7. The nesting argument concerns globally optimized likelihoods; these are local EM fits with a fixed 20-iteration budget. The reported MDL minimum is for this prescribed search.')

    heading(doc,'D13 Interpretation',newpage=True)
    heading(doc,'1 Why the merge initialization gives a better fit',2)
    warm=order['merges'][5]['initial_means']
    para(doc,f'The M = 4 to M = 3 merge initializes means at {vector(warm[0],3)}, {vector(warm[1],3)}, and {vector(warm[2],3)}. These already cover the negative, central positive, and right positive regions. In contrast, seed 2 starts from true classes 1, 1, and 0, and the baseline also starts from 1, 0, and 1; both initially leave the rightmost cluster unrepresented. A component can widen to cover both positive groups while two components stay on the negative group.')
    para(doc,f'The merge fit reaches cost {s["merge3"]["cost"]:.9f}, compared with seed 3’s {runs["seed_3"]["cost"]:.9f}; the difference is only {runs["seed_3"]["cost"]-s["merge3"]["cost"]:.9f}. It is {runs["seed_2"]["cost"]-s["merge3"]["cost"]:.6f} below seed 2. Starting from an already distributed fit explains this improvement, although it does not guarantee the global optimum on every data set.')
    heading(doc,'2 What the spread of final costs means',2)
    para(doc,f'No, the starts do not reach the same final cost. The random-start spread is {max(random_costs)-min(random_costs):.6f}. The high-cost seed-2 fit has a different allocation of components, supporting the presence of different attraction regions on a nonconcave likelihood surface. Smaller cost differences among the good starts also reflect incomplete convergence after 20 iterations; these finite-run results alone do not prove that every distinct endpoint is a distinct local maximum.')
    heading(doc,'3 Why the covariance adjustment prevents collapse',2)
    para(doc,'The weighted covariance is positive semidefinite. Adding εI makes each eigenvalue at least ε = 10⁻⁶, so the determinant is at least εᵖ and a component cannot shrink to a point with unbounded density. This keeps the likelihood finite for the regularized iterates. The cost is biased covariance estimates, reduced ability to represent extremely narrow clusters, and departure from the exact unregularized ML update; the usual strict monotonicity guarantee does not automatically apply to this additive adjustment. The measured decreases here are only rounding-level.')
    heading(doc,'4 What more observations should do to the selected order',2)
    para(doc,'With more observations from this same identifiable three-component distribution, adequate optimization and a suitable candidate-order range should make selection of three more reliable, rather than make the chosen order grow without bound. For p = 2, one extra component costs 3 log(Np) in the criterion. Missing a real component usually loses likelihood on the order of N, which eventually dominates this logarithmic penalty; redundant components should not earn enough improvement to justify their penalty. This is an expected large-sample behavior, not a monotone guarantee for every finite sample or this particular greedy search.')
    heading(doc,'D13 Workflow and specification reflection',newpage=True)
    heading(doc,'5 Advantage and risk of merging from the previous order',2)
    para(doc,'The advantage is an informed initialization: the higher-order fit has already located occupied data regions, and a moment-preserving merge carries that information into the next order. Here that procedure gives a three-component cost essentially equal to the best random start while also selecting the order. The risk is path dependence: a poor higher-order fit or a locally attractive but incorrect merge can steer all subsequent fits into an unfavorable basin. The procedure never revisits a discarded merge, so it can miss a better solution at the same order.')
    heading(doc,'6 Which specifications needed revision',2)
    para(doc,'In this assistant-assisted implementation, none of the seven saved specifications required revision after a failed numerical check. The E-step, M-step, likelihood, and EM checks passed, and the independent audit confirmed the fits and merge choices. This records the actual implementation history; it does not invent a student debugging experience.')
    para(doc,'Several ambiguities were resolved before coding. The run_em contract explicitly includes iteration zero, because the prose return description and the lab pseudocode differ in emphasis. The m_step contract makes the hard-label comparison include the same covariance ridge and uses the ML divisor, not the unbiased sample divisor. The estimate_order contract specifies duplicate iteration coordinates at merges, copied saved models, and precise returned fields. Empty-component behavior and deterministic merge ordering were also specified in advance. These are clarifications of the lab’s short signatures rather than post-failure revisions.')
    heading(doc,'Sources and reproducibility',2)
    p=para(doc,'Assignment source: ');link(p,'Charles A Bouman Lab 6 The EM Algorithm','https://cabouman.github.io/grad_labs/labs-60141/lab6/index.html');p.add_run('. Accessed November 15 2026. Equation numbers and D1–D13 follow that page.')
    para(doc,'All numerical results and figures were generated from the saved sample file in lab6/results/data.npz. The seven specified functions are in lab6.py. Separate scripts generate the data, run experiments, independently verify the numerical results, and create the figures. SPECIFICATIONS.json records the verbatim specifications and WORK_LOG.md records the pre-run prediction and implementation clarifications. No source code is pasted into this report.')
    para(doc,'The lab requests a PDF report and separate plain Python files for its eventual submission. This document remains in editable Word format as requested; export a PDF only after completing your changes. The code and report have been saved locally, with no Brightspace submission or repository push performed.')
    OUT.parent.mkdir(parents=True,exist_ok=True);doc.save(OUT)
    print(OUT)

if __name__=='__main__':main()
