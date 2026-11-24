"""Build an editable Lab 7 Word report with python-docx; never create a PDF."""
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
OUT=HERE/'output/Lab_7_Report.docx'


def element(name,attrs=None):
    value=OxmlElement(name)
    for k,v in (attrs or {}).items():value.set(qn(k),str(v))
    return value


def hyperlink(paragraph,label,url):
    link=element('w:hyperlink',{'r:id':paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True)})
    run=element('w:r');properties=element('w:rPr')
    properties.append(element('w:u',{'w:val':'single'}));run.append(properties)
    text=element('w:t');text.text=label;run.append(text);link.append(run)
    paragraph._p.append(link)


def configure(doc):
    section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.6);section.bottom_margin=Inches(.6)
    section.left_margin=Inches(.7);section.right_margin=Inches(.7)
    section.footer_distance=Inches(.25)
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
        style=doc.styles[name]
        style.font.name='Arial';style.font.color.rgb=RGBColor(0,0,0)
        style.font.size=Pt(10.5)
        style.paragraph_format.space_after=Pt(6)
        style.paragraph_format.line_spacing=1.04
    for name,size in (('Title',22),('Subtitle',13),('Heading 1',13),('Heading 2',11),('Caption',9)):
        doc.styles[name].font.size=Pt(size)
    doc.styles['Title'].font.bold=True
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(8)
    doc.styles['Heading 2'].paragraph_format.space_before=Pt(9)
    doc.styles['Caption'].font.italic=False
    for node in doc.styles.element.xpath('.//w:pBdr'):node.getparent().remove(node)
    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('Lab 7  |  ').font.size=Pt(8)
    field=element('w:fldSimple',{'w:instr':'PAGE'})
    run=element('w:r');text=element('w:t');text.text='1';run.append(text);field.append(run)
    footer._p.append(field)
    doc.core_properties.title='Lab 7 Stochastic Sampling'
    doc.core_properties.author='Vaidehi Pujary'
    doc.core_properties.subject='ECE 60141 GMRF and QGGMRF sampling and posterior uncertainty'


def para(doc,text,style=None):
    return doc.add_paragraph(text,style)


def heading(doc,text,page=False,level=1):
    p=doc.add_heading(text,level)
    if page:p.paragraph_format.page_break_before=True
    return p


def figure(doc,name,width,caption):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.keep_with_next=True
    p.paragraph_format.line_spacing=1
    shape=p.add_run().add_picture(str(RESULTS/name),width=Inches(width))
    shape._inline.docPr.set('descr',caption)
    p=para(doc,caption,'Caption');p.paragraph_format.space_after=Pt(5)


def equation(doc,text):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(6)
    math=element('m:oMath');run=element('m:r');content=element('m:t')
    content.text=text;run.append(content);math.append(run);p._p.append(math)


def main():
    s=json.loads((RESULTS/'summary.json').read_text())
    v=json.loads((RESULTS/'verification.json').read_text())
    posterior=s['posterior'];r=posterior['rmse'];q=s['qggmrf'];c=s['comparison']
    doc=Document();configure(doc)
    para(doc,'Lab 7 Stochastic Sampling','Title')
    para(doc,'Vaidehi Pujary  •  ECE 60141  •  November 24 2026')
    hyperlink(para(doc,'GitHub repository: '),'vaipujary/image-processing-labs','https://github.com/vaipujary/image-processing-labs')
    para(doc,'Seeds: 7 for each prior run and measurement noise; 8 for posterior moments; 9 and 10 for additional chains starting at y and zero. All calculations use float64 PyTorch on CPU, with four sequential parity classes per sweep.')
    para(doc,f'Sampling the denoising posterior gives mean RMSE {r["posterior_mean"]:.5f}, close to MAP at {r["map"]:.5f}, and quantifies uncertainty beyond a point reconstruction.')
    heading(doc,'D1 GMRF prior samples')
    figure(doc,'D1_gmrf_samples.png',6.4,'Figure 1. Zero initialization, 256 × 256 pixels, 200 sweeps; each mean-subtracted image uses its own full display range.')
    para(doc,'Increasing σₓ scales fluctuations linearly: their standard deviations are 0.01606, 0.08030, and 0.32121 at σₓ = 0.01, 0.05, and 0.2; resetting seed 7 makes the textures identical under individual gray scales. The random patches lack coherent object outlines, unlike a photograph.')
    heading(doc,'D2 Brightness drift and centered variation')
    figure(doc,'D2_prior_traces.png',6.6,'Figure 2. Image mean and spatial standard deviation of x − mean(x), after each of 500 sweeps at σₓ = 0.05.')
    para(doc,'The centered standard deviation begins to stabilize, averaging 0.08321 over sweeps 301–400 and 0.08547 over 401–500, while the brightness keeps wandering; 500 sweeps do not establish complete mixing of the slow spatial modes. Equation (1) depends only on differences and leaves the constant brightness mode unpenalized, so the unrestricted prior has no stationary brightness distribution; I subtracted each image’s mean only for display in D1, leaving the sampled chains unchanged.')

    heading(doc,'D3 QGGMRF samples and acceptance rates',page=True)
    para(doc,'Each run starts from zero on a 256 × 256 grid and uses 300 sweeps with σₓ = 0.05, p = 1.2, q = 2, and T = 0.1. Acceptance rates below are averages over all 300 sweeps; the displayed images have their means removed and use individual ranges.')
    figure(doc,'D3_qggmrf_samples.png',7.0,'Figure 3. Final QGGMRF samples, with the mean acceptance fraction printed below each image.')
    heading(doc,'D4 Proposal width and sampling efficiency')
    figure(doc,'D4_proposal_energy.png',7.0,'Figure 4. QGGMRF prior energy after every full sweep; each neighboring pair contributes once.')
    para(doc,f'The best tested width was 0.2: its final 100 sweeps had {q["0.2"]["acceptance_last100"]:.2%} acceptance and mean squared pixel movement {q["0.2"]["mean_squared_jump_last100"]:.6f}, about 13.6 times the movement at width 0.02, while its energy settled near {q["0.2"]["energy_last100_mean"]:.0f}. At 0.002, {q["0.002"]["acceptance_all"]:.2%} acceptance hid extremely slow exploration and an energy still rising at sweep 300; at 0.02, {q["0.02"]["acceptance_all"]:.2%} acceptance also produced highly correlated small moves, so neither smaller width outperformed 0.2 in these measurements. Next time I would start near 40–50% acceptance for these scalar updates and check movement and correlation as well, because the page’s overly large proposal scenario did not occur for the tested width 0.2.')

    heading(doc,'D5 Comparing the two priors',page=True)
    figure(doc,'D5_prior_comparison.png',4.65,f'Figure 5. σₓ = 0.05 for both models; the selected QGGMRF proposal width is 0.2 and both centered images use the common range [{c["display_range"][0]:.4f}, {c["display_range"][1]:.4f}].')
    para(doc,f'The QGGMRF sample has stronger local contrasts and more pronounced jumps than the GMRF sample, with horizontal difference standard deviations {c["qggmrf_difference_std"]:.5f} and {c["gmrf_difference_std"]:.5f}, respectively. For large |Δ|, its potential grows approximately as |Δ|¹·² instead of quadratically, making large differences less costly and explaining the edge preservation that motivated QGGMRF reconstruction in Lab 4.')
    heading(doc,'D6 Posterior denoising and MAP comparison')
    para(doc,'The clean image is the 256 × 256 crop at rows 128–383 and columns 256–511 of kodim23, using zero-based indices; σw = σₓ = 0.05. Posterior moments use 100 discarded sweeps and 400 kept sweeps, and the single sample is sweep 500 of the separate chain initialized at y.')
    figure(doc,'D6_denoising.png',6.8,'Figure 6. The clean crop, noisy measurement, single posterior sample, posterior mean, and MAP estimate all use [0, 1]; values are not clipped for sampling or RMSE.')
    para(doc,f'RMSE against clean: single sample {r["single_sample"]:.6f}; posterior mean {r["posterior_mean"]:.6f}; MAP {r["map"]:.6f}. The MAP result uses the existing Lab 3 gradient descent with A = I for 200 iterations; the maximum absolute gradient is {s["map_gradient_max"]:.2e}.')

    heading(doc,'D7 Pixelwise posterior uncertainty',page=True)
    figure(doc,'D7_uncertainty.png',3.6,f'Figure 7. Empirical posterior standard deviation across 400 kept sweeps, with display range [{posterior["std_range"][0]:.5f}, {posterior["std_range"][1]:.5f}] and variance divisor 400.')
    para(doc,f'The posterior is systematically least certain near the image boundary, especially toward the corners, with mean estimated standard deviation {posterior["std_boundary_mean"]:.5f} on the outer boundary versus {posterior["std_interior_mean"]:.5f} in the interior. Boundary pixels have fewer weighted neighbors to constrain them; for this fixed linear Gaussian model the covariance is (σw⁻²I + σₓ⁻²L)⁻¹, independent of the observed intensities, so apparent interior speckle is largely Monte Carlo variation rather than evidence of higher uncertainty at object edges.')
    heading(doc,'D8 Checking burn in from two starting images')
    figure(doc,'D8_burnin.png',7.0,'Figure 8. Two independent 500-sweep posterior chains; the left panel includes the initial energy transient and the right zooms into the stationary band over all 500 sweeps.')
    para(doc,f'The energy traces enter the same fluctuation band after approximately 4–5 sweeps, with post-burn-in means {posterior["energy_y_last400_mean"]:.1f} from y and {posterior["energy_zero_last400_mean"]:.1f} from zero, compared with within-chain standard deviations {posterior["energy_y_last400_std"]:.1f} and {posterior["energy_zero_last400_std"]:.1f}. Agreement after deliberately different initial conditions is stronger evidence of loss of initialization bias than one apparently flat trace, although energy agreement alone does not prove convergence of every statistic. The 100-sweep burn-in is therefore conservative for this experiment, and the data term also anchors brightness: after burn-in the image mean stays within [{posterior["posterior_mean_range_after100"][0]:.6f}, {posterior["posterior_mean_range_after100"][1]:.6f}] around the measured mean {posterior["data_mean"]:.6f}, unlike the prior’s wandering mean in D2.')

    heading(doc,'D9 Discussion',page=True)
    heading(doc,'1 A posterior sample and the MAP estimate',level=2)
    para(doc,f'The sample is visibly rougher: the root mean square horizontal difference is {posterior["roughness"]["single_sample"]:.5f}, compared with {posterior["roughness"]["map"]:.5f} for MAP, and its reconstruction RMSE is also higher ({r["single_sample"]:.5f} versus {r["map"]:.5f}). MAP selects the image of maximum posterior density, whereas a draw includes random variation around that image and typically lies in the much larger volume away from the mode. Thus the prompt’s suggestion that the rougher image has the lower RMSE is not supported by this run; maximizing density and minimizing realized error against the particular clean image are different criteria.')
    heading(doc,'2 Why averaging samples improves the estimate',level=2)
    para(doc,f'Averaging suppresses the posterior fluctuations present in a single draw, reducing RMSE here from {r["single_sample"]:.5f} to {r["posterior_mean"]:.5f}. By the squared-error estimator result in Lecture 4, E[X | y] minimizes E[‖X − a‖² | y] over estimates a, making it the conditional minimum mean square error estimator. This Gaussian posterior has its exact mean at the MAP estimate, so the {posterior["mean_map_rmse"]:.5f} RMSE between the finite 400-sweep mean and MAP is Monte Carlo error, and the sample average need not beat exact MAP on this clean image.')
    heading(doc,'3 Gibbs as a Hastings Metropolis update',level=2)
    para(doc,'Gibbs proposes the new coordinate from its exact full conditional, q(wₛ | x) = p(wₛ | x₋ₛ), keeping every other coordinate fixed. In the Hastings ratio the conditional density ratio of the target cancels the reverse-to-forward proposal ratio, giving acceptance exactly one:')
    equation(doc,'α = min{1, [p(wₛ | x₋ₛ) / p(xₛ | x₋ₛ)] [p(xₛ | x₋ₛ) / p(wₛ | x₋ₛ)]} = 1')
    para(doc,'Equation (6) on the lab page is the symmetric-proposal specialization; a Gibbs proposal is generally asymmetric, so inserting only its energy change into that simplified formula would omit the cancellation.')
    heading(doc,'4 Why a two color checkerboard fails',level=2)
    para(doc,'Pixels (0, 0) and (1, 1) have the same checkerboard color, but they are diagonal neighbors with weight 1/12. Drawing both independently from conditionals evaluated at their old values is not a draw from their joint conditional because they interact. Consequently the stationary neighboring-pixel correlations, and generally the sampled covariance and variances, would be wrong; four parity classes exclude this diagonal conflict.')
    heading(doc,'5 A practical use of an uncertainty image',level=2)
    para(doc,'A materials scientist could use the uncertainty map to select a region for additional measurements when a reconstructed pore boundary is poorly constrained. Two regions with similar reconstructed intensity can have different uncertainty, so the reconstruction alone cannot show where extra observations are most useful. Here the interpretation is limited to uncertainty under the stated Gaussian noise and GMRF assumptions, rather than uncertainty from an incorrect physical model.')
    heading(doc,'Source and reproducibility',level=2)
    hyperlink(para(doc,'Assignment: '),'Lab 7 Stochastic Sampling','https://cabouman.github.io/grad_labs/labs-60141/lab7/index.html')
    para(doc,'Code and measured results are in lab7.py, run_experiments.py, and results/. All six required signatures, boundary weights, local energy changes, and moment accumulation passed checks; an independent 3 × 4 Gaussian posterior check gave covariance relative error '+f'{v["small_posterior_covariance_relative_frobenius_error"]:.2%}'+'. This editable report was prepared with AI assistance; its plots can be regenerated from make_figures.py.')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__=='__main__':main()
