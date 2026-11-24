"""Check DOCX package, required content, and image geometry without rendering."""
import json
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from PIL import Image

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'output/Lab_7_Report.docx'
    doc=Document(path)
    paragraphs=[p.text for p in doc.paragraphs]
    headings=[p.text for p in doc.paragraphs if p.style.name=='Heading 1']
    assert [x.split()[0] for x in headings]==[f'D{i}' for i in range(1,10)]
    text='\n'.join(paragraphs)
    assert 'Vaidehi Pujary' in text and 'Seeds: 7' in text
    assert len(doc.inline_shapes)==8
    section=doc.sections[0]
    usable=section.page_width-section.left_margin-section.right_margin
    assert all(shape.width<=usable for shape in doc.inline_shapes)
    assert all(shape.height<section.page_height-section.top_margin-section.bottom_margin for shape in doc.inline_shapes)
    assert all(shape._inline.docPr.get('descr') for shape in doc.inline_shapes)
    assert sum(p.paragraph_format.page_break_before is True for p in doc.paragraphs)==4
    assert len(doc._element.xpath('.//m:oMath'))==1
    assert not doc._element.xpath('.//w:documentProtection')
    assert not doc._element.xpath('.//w:pBdr')
    for label in ('1 A posterior sample','2 Why averaging','3 Gibbs as','4 Why a two','5 A practical'):
        assert any(p.startswith(label) for p in paragraphs)
    summary=json.loads((HERE/'results/summary.json').read_text())
    for key in ('single_sample','posterior_mean','map'):
        assert f'{summary["posterior"]["rmse"][key]:.6f}' in text
    assert not list(HERE.rglob('*.pdf'))
    with ZipFile(path) as z:
        assert z.testzip() is None
        media=[name for name in z.namelist() if name.startswith('word/media/')]
        assert len(media)==8
        for name in media:
            with z.open(name) as file:
                with Image.open(file) as image:
                    assert image.width>=800 and image.height>=500
                    image.verify()
        xml=z.read('word/_rels/document.xml.rels').decode()
        assert 'https://github.com/vaipujary/image-processing-labs' in xml
        assert 'https://cabouman.github.io/grad_labs/labs-60141/lab7/index.html' in xml
    result={'docx_package_integrity':True,'D1_through_D9_in_order':True,
            'figures':8,'five_discussion_answers':True,'editable_text_and_equation':True,
            'image_geometry_within_page':True,'rmse_matches_measured_results':True,
            'no_pdf_created':True,'visual_render_review':False,
            'review_note':'File-based checks only, as requested; no screenshots or PDF conversion.'}
    (HERE/'results/report_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
