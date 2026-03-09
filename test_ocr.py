import fitz

doc = fitz.open("data/SEBI _ Master Circular on the redressal of investor grievances through the SEBI Complaints Redress System (SCORES) platform.pdf")
print("Total pages:", len(doc))
text = ""
for page in doc:
    text += page.get_text()
    
if not text.strip():
    print("NO TEXT LAYER FOUND. This seems to be an image-only PDF.")
